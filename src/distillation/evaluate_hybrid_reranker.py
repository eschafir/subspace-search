import os
import sys
import json
import argparse
import random
import torch
import torch.nn.functional as F
import faiss
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

# Add project root parent to sys.path to find src
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.models import load, best_gpu
from src.distillation.evaluate_hybrid import clean_query_text, min_max_normalize

def evaluate_split_hybrid_reranker(model, tokenizer, query_lut, dense_model, dense_processor, is_clip,
                                   reranker_model, reranker_tokenizer, device, dataset_name, split_name,
                                   batch_size=32, dense_batch_size=256, logit_threshold=0.0,
                                   filter_stopwords=False, beta=0.6, rerank_top_k=30):
    """Evaluate V-SPLADE student + Dense + FAISS retrieval followed by Cross-Encoder reranking on a single split."""
    print(f"\nEvaluating split '{split_name}' with Reranker...")
    print(f"Params: beta={beta}, rerank_top_k={rerank_top_k}")
    
    # Load dataset split docs & examples
    try:
        docs_ds = load_dataset(dataset_name, "documents", split=split_name)
        queries_ds = load_dataset(dataset_name, "examples", split=split_name)
    except Exception as e:
        print(f"Error loading split '{split_name}': {e}")
        return None
        
    print(f"Loaded {len(docs_ds)} documents and {len(queries_ds)} queries.")
    
    corpus_ids = []
    corpus_texts = {}
    w_p_list = []
    dense_embeds_list = []
    
    model.eval()
    dense_model.eval()
    reranker_model.eval()
    vocab_size = model.config.vocab_size
    
    # --- 1. Encode all document corpus once ---
    print("Encoding corpus documents (V-SPLADE and Dense)...")
    with torch.no_grad():
        for i in tqdm(range(0, len(docs_ds), dense_batch_size), desc="Doc Batches"):
            batch = docs_ds[i : i + dense_batch_size]
            doc_texts = batch.get("content", batch.get("text", []))
            doc_texts = [text if text is not None else "[No text]" for text in doc_texts]
            
            doc_ids = batch.get("id", [])
            corpus_ids.extend(doc_ids)
            
            for d_id, text in zip(doc_ids, doc_texts):
                corpus_texts[d_id] = text
                
            # V-SPLADE encoding
            v_sub_batch = batch_size
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
                if logit_threshold > 0.0:
                    z_p = z_p - logit_threshold
                w_p = torch.log1p(torch.relu(z_p))
                w_p_list.append(w_p.cpu())
                
            # Dense encoding
            if is_clip:
                clip_inputs = dense_processor(
                    text=doc_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77
                ).to(device)
                doc_features = dense_model.get_text_features(**clip_inputs)
                if hasattr(doc_features, "text_embeds"):
                    doc_features = doc_features.text_embeds
                elif hasattr(doc_features, "pooler_output"):
                    doc_features = doc_features.pooler_output
                elif not isinstance(doc_features, torch.Tensor) and hasattr(doc_features, "last_hidden_state"):
                    doc_features = doc_features.last_hidden_state
                    
                doc_features = doc_features / doc_features.norm(p=2, dim=-1, keepdim=True)
            else:
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
            
    # Concatenate representations
    w_p_all = torch.cat(w_p_list, dim=0) # (num_docs, vocab_size)
    dense_embeds_all = np.vstack(dense_embeds_list) # (num_docs, dense_dim)
    
    # --- 2. Build FAISS Index ---
    print("Building FAISS index for dense representations...")
    dimension = dense_embeds_all.shape[1]
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(dense_embeds_all)
    print("FAISS index built successfully.")
    
    # Evaluate queries
    mrr_total = 0.0
    ndcg_total = 0.0
    count = 0
    
    print("Evaluating queries...")
    for row in tqdm(queries_ds, desc="Queries"):
        q_id = row.get("id")
        q_text = row.get("query")
        gold_ids = row.get("gold_ids", [])
        if not gold_ids:
            continue
            
        if filter_stopwords:
            q_text_cleaned = clean_query_text(q_text)
        else:
            q_text_cleaned = q_text
            
        # --- Stage 1: Retrieval ---
        # A. Sparse scores
        query_token_ids = tokenizer(q_text_cleaned, add_special_tokens=False)["input_ids"]
        if not query_token_ids:
            continue
            
        w_q = torch.zeros(vocab_size, dtype=torch.float32)
        with torch.no_grad():
            w_q_weights = F.softplus(query_lut[query_token_ids]).cpu()
            w_q[query_token_ids] = w_q_weights
        scores_sparse = torch.sum(w_q.unsqueeze(0) * w_p_all, dim=1).to(device)
        
        # B. Dense scores
        with torch.no_grad():
            if is_clip:
                q_inputs = dense_processor(
                    text=[q_text],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77
                ).to(device)
                q_features = dense_model.get_text_features(**q_inputs)
                if hasattr(q_features, "text_embeds"):
                    q_features = q_features.text_embeds
                elif hasattr(q_features, "pooler_output"):
                    q_features = q_features.pooler_output
                elif not isinstance(q_features, torch.Tensor) and hasattr(q_features, "last_hidden_state"):
                    q_features = q_features.last_hidden_state
                q_features = q_features / q_features.norm(p=2, dim=-1, keepdim=True)
            else:
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
            
        # C. Score Fusion
        norm_sparse = min_max_normalize(scores_sparse)
        norm_dense = min_max_normalize(scores_dense)
        scores_hybrid = (1.0 - beta) * norm_sparse + beta * norm_dense
        
        # D. Get Stage 1 Ranked Candidates
        ranked_indices = torch.argsort(scores_hybrid, descending=True).tolist()
        ranked_doc_ids = [corpus_ids[idx] for idx in ranked_indices]
        
        # --- Stage 2: Cross-Encoder Reranking ---
        # Select top-K candidates to rerank
        candidates_to_rerank = ranked_doc_ids[:rerank_top_k]
        candidate_texts = [corpus_texts.get(d_id, "[No text]") for d_id in candidates_to_rerank]
        
        # Construct query-document pairs
        pairs = [[q_text, doc_text] for doc_text in candidate_texts]
        
        # Tokenize and score pairs
        rerank_inputs = reranker_tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            rerank_outputs = reranker_model(**rerank_inputs)
            # Cross-encoder logits represent similarity scores
            rerank_scores = rerank_outputs.logits.view(-1).tolist()
            
        # Re-sort candidates based on cross-encoder logits
        scored_candidates = list(zip(candidates_to_rerank, rerank_scores))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        reranked_top_ids = [item[0] for item in scored_candidates]
        
        # Construct the final hybrid list: replace the top K with reranked list
        final_doc_ids = reranked_top_ids + ranked_doc_ids[rerank_top_k:]
        
        # Calculate MRR@10
        mrr_val = 0.0
        for rank_idx, doc_id in enumerate(final_doc_ids[:10]):
            if doc_id in gold_ids:
                mrr_val = 1.0 / (rank_idx + 1)
                break
        mrr_total += mrr_val
        
        # Calculate NDCG@10
        dcg = 0.0
        for rank_idx, doc_id in enumerate(final_doc_ids[:10]):
            if doc_id in gold_ids:
                dcg += 1.0 / (torch.log2(torch.tensor(rank_idx + 2.0)).item())
                
        idcg = 0.0
        for rank_idx in range(min(10, len(gold_ids))):
            idcg += 1.0 / (torch.log2(torch.tensor(rank_idx + 2.0)).item())
            
        ndcg_val = dcg / idcg if idcg > 0.0 else 0.0
        ndcg_total += ndcg_val
        count += 1
        
    if count == 0:
        return 0.0, 0.0
        
    mrr = mrr_total / count
    ndcg = ndcg_total / count
    print(f"Reranked Results for split '{split_name}': MRR@10 = {mrr:.4f}, NDCG@10 = {ndcg:.4f}")
    return mrr, ndcg

def main():
    parser = argparse.ArgumentParser(description="HIVE-to-V-SPLADE Hybrid Retrieval + Cross-Encoder Reranking")
    parser.add_argument("--checkpoint-path", type=str, default="results/vsplade_student_checkpoint.pt", help="Path to distilled student weights")
    parser.add_argument("--model", type=str, default="qwen-1.5b", help="Model key of the student backbone")
    parser.add_argument("--dense-model", type=str, default="BAAI/bge-small-en-v1.5", help="Pretrained dense model key")
    parser.add_argument("--reranker-model", type=str, default="BAAI/bge-reranker-base", help="Pretrained cross-encoder reranker model key")
    parser.add_argument("--dataset", type=str, default="mm-bright/MM-BRIGHT", help="Dataset name on Hugging Face")
    parser.add_argument("--test-splits", type=str, default="law,quant,quantumcomputing,robotics", help="Comma-separated splits to evaluate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for V-SPLADE document encoding")
    parser.add_argument("--dense-batch-size", type=int, default=256, help="Batch size for Dense model document encoding")
    parser.add_argument("--logit-threshold", type=float, default=0.0, help="Logit threshold to enforce sparsity")
    parser.add_argument("--filter-stopwords", action="store_true", help="Clean queries before tokenization")
    parser.add_argument("--beta", type=float, default=0.6, help="Optimal score-fusion weight found in sweep")
    parser.add_argument("--rerank-top-k", type=int, default=30, help="Number of retrieved candidates to pass to cross-encoder")
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
    dense_model_name = args.dense_model
    is_clip = "clip" in dense_model_name.lower()
    print(f"Loading dense model '{dense_model_name}' on {device} (is_clip = {is_clip})...")
    if is_clip:
        from transformers import CLIPProcessor, CLIPModel
        dense_model = CLIPModel.from_pretrained(dense_model_name).to(device)
        dense_processor = CLIPProcessor.from_pretrained(dense_model_name)
    else:
        dense_model = AutoModel.from_pretrained(dense_model_name).to(device)
        dense_processor = AutoTokenizer.from_pretrained(dense_model_name)
        
    # 3. Load Cross-Encoder Reranker
    print(f"Loading cross-encoder reranker '{args.reranker_model}' on {device}...")
    reranker_model = AutoModelForSequenceClassification.from_pretrained(args.reranker_model).to(device)
    reranker_tokenizer = AutoTokenizer.from_pretrained(args.reranker_model)
    print("All models loaded successfully.")
    
    # Parse test splits
    test_splits = [s.strip() for s in args.test_splits.split(",") if s.strip()]
    
    all_results = {}
    mrr_sum = 0.0
    ndcg_sum = 0.0
    evaluated_count = 0
    
    for split in test_splits:
        res = evaluate_split_hybrid_reranker(
            model, tokenizer, query_lut, dense_model, dense_processor, is_clip,
            reranker_model, reranker_tokenizer, device, args.dataset, split,
            batch_size=args.batch_size, dense_batch_size=args.dense_batch_size,
            logit_threshold=args.logit_threshold, filter_stopwords=args.filter_stopwords,
            beta=args.beta, rerank_top_k=args.rerank_top_k
        )
        if res:
            mrr, ndcg = res
            all_results[split] = {"MRR@10": mrr, "NDCG@10": ndcg}
            mrr_sum += mrr
            ndcg_sum += ndcg
            evaluated_count += 1
            
    if evaluated_count > 0:
        print("\n==========================================")
        print("Summary of Zero-Shot Hybrid + Reranker Evaluation:")
        print("==========================================\n")
        for split, metrics in all_results.items():
            print(f"Split: {split:<20} | MRR@10 = {metrics['MRR@10']:.4f} | NDCG@10 = {metrics['NDCG@10']:.4f}")
        print("-" * 50)
        print(f"AVERAGE:                  | MRR@10 = {mrr_sum/evaluated_count:.4f} | NDCG@10 = {ndcg_sum/evaluated_count:.4f}")
        print("==========================================\n")

if __name__ == "__main__":
    main()
