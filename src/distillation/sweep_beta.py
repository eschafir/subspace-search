import os
import sys
import json
import argparse
import torch
import torch.nn.functional as F
import faiss
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

# Add project root parent to sys.path to find src
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.models import load, best_gpu
from src.distillation.evaluate_hybrid import clean_query_text, min_max_normalize

def main():
    parser = argparse.ArgumentParser(description="V-SPLADE + BGE Fast Hyperparameter Beta Sweep")
    parser.add_argument("--checkpoint-path", type=str, default="results/vsplade_student_checkpoint.pt", help="Path to distilled student weights")
    parser.add_argument("--model", type=str, default="qwen-1.5b", help="Model key of the student backbone")
    parser.add_argument("--dense-model", type=str, default="BAAI/bge-small-en-v1.5", help="Pretrained dense model key")
    parser.add_argument("--dataset", type=str, default="mm-bright/MM-BRIGHT", help="Dataset name on Hugging Face")
    parser.add_argument("--test-splits", type=str, default="law,quant,quantumcomputing,robotics", help="Comma-separated splits to sweep")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for V-SPLADE document encoding")
    parser.add_argument("--dense-batch-size", type=int, default=256, help="Batch size for Dense model document encoding")
    parser.add_argument("--logit-threshold", type=float, default=8.0, help="Logit threshold to enforce sparsity")
    parser.add_argument("--filter-stopwords", action="store_true", default=True, help="Clean queries before tokenization")
    args = parser.parse_args()
    
    device = best_gpu()
    
    # 1. Load V-SPLADE Student
    print(f"Loading base student model '{args.model}' on {device}...")
    model, tokenizer = load(args.model, device=device)
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint file not found at {args.checkpoint_path}.")
        return
    print(f"Loading checkpoint weights from {args.checkpoint_path}...")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    query_lut = checkpoint["query_lut"].to(device)
    
    # 2. Load Dense Model
    print(f"Loading dense model '{args.dense_model}' on {device}...")
    dense_model = AutoModel.from_pretrained(args.dense_model).to(device)
    dense_processor = AutoTokenizer.from_pretrained(args.dense_model)
    print("Models loaded successfully.")
    
    # Parse splits
    splits = [s.strip() for s in args.test_splits.split(",") if s.strip()]
    
    # We will sweep beta values from 0.0 to 1.0
    beta_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    model.eval()
    dense_model.eval()
    vocab_size = model.config.vocab_size
    
    overall_results = {}
    
    for split_name in splits:
        print(f"\n==========================================")
        print(f"Processing split: '{split_name}'")
        print(f"==========================================")
        
        # Load dataset split docs & examples
        try:
            docs_ds = load_dataset(args.dataset, "documents", split=split_name)
            queries_ds = load_dataset(args.dataset, "examples", split=split_name)
        except Exception as e:
            print(f"Error loading split '{split_name}': {e}")
            continue
            
        corpus_ids = []
        w_p_list = []
        dense_embeds_list = []
        
        # 1. Encode all document corpus once
        print("Encoding corpus documents (V-SPLADE and Dense)...")
        with torch.no_grad():
            for i in tqdm(range(0, len(docs_ds), args.dense_batch_size), desc="Doc Batches"):
                batch = docs_ds[i : i + args.dense_batch_size]
                doc_texts = batch.get("content", batch.get("text", []))
                doc_texts = [text if text is not None else "[No text]" for text in doc_texts]
                
                doc_ids = batch.get("id", [])
                corpus_ids.extend(doc_ids)
                
                # V-SPLADE encoding
                v_sub_batch = args.batch_size
                for sj in range(0, len(doc_texts), v_sub_batch):
                    sub_texts = doc_texts[sj : sj + v_sub_batch]
                    doc_inputs = tokenizer(
                        sub_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512
                    ).to(device)
                    
                    outputs = model.model(
                        input_ids=doc_inputs["input_ids"],
                        attention_mask=doc_inputs["attention_mask"],
                        output_hidden_states=True
                    )
                    hidden_states = outputs.hidden_states[-1]
                    
                    attention_mask = doc_inputs["attention_mask"].unsqueeze(-1)
                    chunk_size = 20000
                    z_p_parts = []
                    for start in range(0, vocab_size, chunk_size):
                        weight_chunk = model.lm_head.weight[start : start + chunk_size]
                        logits_chunk = torch.matmul(hidden_states, weight_chunk.t())
                        logits_chunk = logits_chunk * attention_mask + (1 - attention_mask) * -1e9
                        z_p_chunk, _ = torch.max(logits_chunk, dim=1)
                        z_p_parts.append(z_p_chunk)
                        
                    z_p = torch.cat(z_p_parts, dim=1)
                    if args.logit_threshold > 0.0:
                        z_p = z_p - args.logit_threshold
                    w_p = torch.log1p(torch.relu(z_p))
                    w_p_list.append(w_p.cpu())
                    
                # Dense encoding
                inputs = dense_processor(
                    doc_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(device)
                outputs = dense_model(**inputs)
                
                token_embeddings = outputs[0]
                attention_mask = inputs['attention_mask'].unsqueeze(-1)
                input_mask_expanded = attention_mask.expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                doc_features = sum_embeddings / sum_mask
                doc_features = F.normalize(doc_features, p=2, dim=1)
                
                dense_embeds_list.append(doc_features.cpu().numpy().astype('float32'))
                
        w_p_all = torch.cat(w_p_list, dim=0) # (num_docs, vocab_size)
        dense_embeds_all = np.vstack(dense_embeds_list) # (num_docs, dense_dim)
        
        # Build FAISS index
        print("Building FAISS index for dense representations...")
        dimension = dense_embeds_all.shape[1]
        faiss_index = faiss.IndexFlatIP(dimension)
        faiss_index.add(dense_embeds_all)
        
        # 2. Compute and cache raw scores for all queries once
        print("Pre-calculating and caching raw query search scores...")
        cached_query_scores = []
        
        for row in tqdm(queries_ds, desc="Queries"):
            q_id = row.get("id")
            q_text = row.get("query")
            gold_ids = row.get("gold_ids", [])
            if not gold_ids:
                continue
                
            if args.filter_stopwords:
                q_text_cleaned = clean_query_text(q_text)
            else:
                q_text_cleaned = q_text
                
            # V-SPLADE sparse score
            query_token_ids = tokenizer(q_text_cleaned, add_special_tokens=False)["input_ids"]
            if not query_token_ids:
                continue
                
            w_q = torch.zeros(vocab_size, dtype=torch.float32)
            with torch.no_grad():
                w_q_weights = F.softplus(query_lut[query_token_ids]).cpu()
                w_q[query_token_ids] = w_q_weights
            scores_sparse = torch.sum(w_q.unsqueeze(0) * w_p_all, dim=1).to(device)
            
            # Dense score
            with torch.no_grad():
                q_inputs = dense_processor(
                    [q_text],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(device)
                q_outputs = dense_model(**q_inputs)
                
                token_embeddings = q_outputs[0]
                attention_mask = q_inputs['attention_mask'].unsqueeze(-1)
                input_mask_expanded = attention_mask.expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                q_features = sum_embeddings / sum_mask
                q_features = F.normalize(q_features, p=2, dim=1)
                
                q_embed_np = q_features.cpu().numpy().astype('float32')
                
            D, I = faiss_index.search(q_embed_np, len(corpus_ids))
            
            scores_dense = torch.zeros(len(corpus_ids), device=device)
            for rank_idx, doc_idx in enumerate(I[0]):
                scores_dense[doc_idx] = float(D[0][rank_idx])
                
            cached_query_scores.append((scores_sparse, scores_dense, gold_ids))
            
        # 3. Sweep Beta in-memory (virtually instantaneous!)
        split_results = []
        progress_bar = tqdm(beta_values, desc=f"Sweeping Beta ({split_name})")
        for beta in progress_bar:
            mrr_total = 0.0
            ndcg_total = 0.0
            count = 0
            
            for scores_sparse, scores_dense, gold_ids in cached_query_scores:
                norm_sparse = min_max_normalize(scores_sparse)
                norm_dense = min_max_normalize(scores_dense)
                scores_hybrid = (1.0 - beta) * norm_sparse + beta * norm_dense
                
                ranked_indices = torch.argsort(scores_hybrid, descending=True).tolist()
                ranked_doc_ids = [corpus_ids[idx] for idx in ranked_indices]
                
                # MRR@10
                mrr_val = 0.0
                for rank_idx, doc_id in enumerate(ranked_doc_ids[:10]):
                    if doc_id in gold_ids:
                        mrr_val = 1.0 / (rank_idx + 1)
                        break
                mrr_total += mrr_val
                
                # NDCG@10
                dcg = 0.0
                for rank_idx, doc_id in enumerate(ranked_doc_ids[:10]):
                    if doc_id in gold_ids:
                        dcg += 1.0 / (torch.log2(torch.tensor(rank_idx + 2.0)).item())
                        
                idcg = 0.0
                for rank_idx in range(min(10, len(gold_ids))):
                    idcg += 1.0 / (torch.log2(torch.tensor(rank_idx + 2.0)).item())
                    
                ndcg_val = dcg / idcg if idcg > 0.0 else 0.0
                ndcg_total += ndcg_val
                count += 1
                
            mrr = mrr_total / count if count > 0 else 0.0
            ndcg = ndcg_total / count if count > 0 else 0.0
            split_results.append((beta, mrr, ndcg))
            progress_bar.set_postfix(beta=f"{beta:.1f}", MRR=f"{mrr:.4f}")
            
        overall_results[split_name] = split_results
        
    # Print final summary table
    print("\n==========================================")
    print("FINAL HYPERPARAMETER BETA SWEEP SUMMARY")
    print("==========================================")
    for beta_idx, beta in enumerate(beta_values):
        print(f"\n--- Beta = {beta:.1f} ---")
        mrr_sum = 0.0
        ndcg_sum = 0.0
        active_splits = 0
        for split_name in splits:
            if split_name in overall_results:
                b, mrr, ndcg = overall_results[split_name][beta_idx]
                print(f"  Split: {split_name:<20} | MRR@10 = {mrr:.4f} | NDCG@10 = {ndcg:.4f}")
                mrr_sum += mrr
                ndcg_sum += ndcg
                active_splits += 1
        if active_splits > 0:
            print(f"  AVERAGE:                  | MRR@10 = {mrr_sum/active_splits:.4f} | NDCG@10 = {ndcg_sum/active_splits:.4f}")
            
if __name__ == "__main__":
    main()
