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
from transformers import AutoTokenizer, AutoModel

# Add project root parent to sys.path to find src
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.models import load, best_gpu

# Common English stop words
stop_words = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
    'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up',
    'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
    'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
    'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
    'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'could', 'didn', 'doesn', 'hadn',
    'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'
])

def clean_query_text(text):
    """Remove HTML, punctuation, and common English stop-words from queries before tokenization."""
    import re
    text = re.sub(r'<[^>]+>', '', text)
    text = ''.join(c for c in text if c.isalnum() or c.isspace())
    words = text.split()
    filtered = [w for w in words if w.lower() not in stop_words]
    return ' '.join(filtered)

def min_max_normalize(scores):
    """Min-max normalize a 1D tensor to [0, 1] range."""
    s_min = scores.min()
    s_max = scores.max()
    if s_max > s_min:
        return (scores - s_min) / (s_max - s_min)
    return torch.zeros_like(scores)

def load_colbert_projection(model_name, device):
    """Load the ColBERT projection layer (linear.weight) from HF checkpoint if available."""
    try:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        try:
            weights_file = hf_hub_download(repo_id=model_name, filename='model.safetensors')
            state = load_file(weights_file)
        except Exception:
            weights_file = hf_hub_download(repo_id=model_name, filename='pytorch_model.bin')
            state = torch.load(weights_file, map_location="cpu")
            
        if 'linear.weight' in state:
            print("Loaded ColBERT linear projection weights from checkpoint.")
            return state['linear.weight'].to(device)
    except Exception as e:
        pass
    return None

def evaluate_split_hybrid(model, tokenizer, query_lut, dense_model, dense_processor, is_clip, device, 
                          dataset_name, split_name, batch_size=32, dense_batch_size=256, 
                          logit_threshold=0.0, filter_stopwords=False, beta=0.4, dense_mode="bi-encoder"):
    """Evaluate hybrid V-SPLADE student + dense (CLIP / Sentence-Transformers) retrieval on a single split."""
    print(f"\nEvaluating split: '{split_name}' with beta = {beta} (is_clip = {is_clip})...")
    
    # Load dataset split docs & examples
    try:
        docs_ds = load_dataset(dataset_name, "documents", split=split_name)
        queries_ds = load_dataset(dataset_name, "examples", split=split_name)
    except Exception as e:
        print(f"Error loading split '{split_name}': {e}")
        return None
        
    print(f"Loaded {len(docs_ds)} documents and {len(queries_ds)} queries.")
    
    corpus_ids = []
    w_p_list = []
    dense_embeds_list = []
    
    model.eval()
    dense_model.eval()
    vocab_size = model.config.vocab_size
    colbert_proj = load_colbert_projection(dense_model.config._name_or_path, device)
    
    print("Encoding corpus documents (V-SPLADE and Dense)...")
    with torch.no_grad():
        for i in tqdm(range(0, len(docs_ds), dense_batch_size), desc="Doc Batches"):
            batch = docs_ds[i : i + dense_batch_size]
            doc_texts = batch.get("content", batch.get("text", []))
            doc_texts = [text if text is not None else "[No text]" for text in doc_texts]
            
            doc_ids = batch.get("id", [])
            corpus_ids.extend(doc_ids)
            
            # --- 1. V-SPLADE encoding ---
            # Process in smaller sub-batches to prevent OOM
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
                
                attention_mask = doc_inputs["attention_mask"].to(hidden_states.device).unsqueeze(-1)
                chunk_size = 20000
                z_p_parts = []
                for start in range(0, vocab_size, chunk_size):
                    weight_chunk = model.lm_head.weight[start : start + chunk_size].to(hidden_states.device)
                    logits_chunk = torch.matmul(hidden_states, weight_chunk.t())
                    logits_chunk = logits_chunk * attention_mask + (1 - attention_mask) * -1e9
                    z_p_chunk, _ = torch.max(logits_chunk, dim=1)
                    z_p_parts.append(z_p_chunk)
                    
                z_p = torch.cat(z_p_parts, dim=1)
                if logit_threshold > 0.0:
                    z_p = z_p - logit_threshold
                w_p = torch.log1p(torch.relu(z_p))
                w_p_list.append(w_p.cpu())
                
            # --- 2. Dense encoding ---
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
                # Sentence-Transformers path via AutoModel (mean pooling or late interaction)
                inputs = dense_processor(
                    doc_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(device)
                outputs = dense_model(**inputs)
                
                token_embeddings = outputs[0]
                if colbert_proj is not None:
                    token_embeddings = torch.matmul(token_embeddings, colbert_proj.t())
                if dense_mode == "late-interaction":
                    norm_tokens = F.normalize(token_embeddings, p=2, dim=-1)
                    for b_idx in range(len(doc_texts)):
                        mask = inputs['attention_mask'][b_idx] == 1
                        valid_tokens = norm_tokens[b_idx][mask]
                        dense_embeds_list.append(valid_tokens.cpu())
                else:
                    attention_mask = inputs['attention_mask'].to(hidden_states.device).unsqueeze(-1)
                    input_mask_expanded = attention_mask.expand(token_embeddings.size()).float()
                    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    doc_features = sum_embeddings / sum_mask
                    doc_features = F.normalize(doc_features, p=2, dim=1)
                    dense_embeds_list.append(doc_features.cpu().numpy().astype('float32'))
            
    # Concatenate representations
    w_p_all = torch.cat(w_p_list, dim=0) # (num_docs, vocab_size)
    
    # --- 3. Build Dense Index / Representations ---
    if dense_mode == "late-interaction":
        print("Preparing late-interaction corpus representations...")
        # Keep corpus representations on CPU to prevent CUDA OOMs
        corpus_embeds = torch.nn.utils.rnn.pad_sequence(dense_embeds_list, batch_first=True, padding_value=0.0) # (num_docs, max_len, dense_dim)
        doc_lengths = [t.shape[0] for t in dense_embeds_list]
        max_len = corpus_embeds.shape[1]
        corpus_masks = torch.zeros(len(dense_embeds_list), max_len) # Keep mask on CPU
        for idx, l in enumerate(doc_lengths):
            corpus_masks[idx, :l] = 1.0
        faiss_index = None
    else:
        dense_embeds_all = np.vstack(dense_embeds_list) # (num_docs, dense_dim)
        print("Building FAISS index for dense representations...")
        dimension = dense_embeds_all.shape[1]
        faiss_index = faiss.IndexFlatIP(dimension) # Flat Inner-Product index for cosine similarity
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
            
        # Clean query text at the string level if requested
        if filter_stopwords:
            q_text_cleaned = clean_query_text(q_text)
        else:
            q_text_cleaned = q_text
            
        # --- 1. Compute V-SPLADE Student Scores ---
        query_token_ids = tokenizer(q_text_cleaned, add_special_tokens=False)["input_ids"]
        if not query_token_ids:
            continue
            
        w_q = torch.zeros(vocab_size, dtype=torch.float32)
        with torch.no_grad():
            w_q_weights = F.softplus(query_lut[query_token_ids]).cpu()
            w_q[query_token_ids] = w_q_weights
        scores_sparse = torch.sum(w_q.unsqueeze(0) * w_p_all, dim=1).to(device) # (num_docs,)
        
        # --- 2. Compute Dense Scores via FAISS or Batched Late-Interaction ---
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
                if colbert_proj is not None:
                    token_embeddings = torch.matmul(token_embeddings, colbert_proj.t())
                if dense_mode == "late-interaction":
                    q_norm_tokens = F.normalize(token_embeddings, p=2, dim=-1)
                    q_mask = q_inputs['attention_mask'][0] == 1
                    q_embeds = q_norm_tokens[0][q_mask] # shape: [L_q, D] (device: GPU)
                else:
                    attention_mask = q_inputs['attention_mask'].unsqueeze(-1)
                    input_mask_expanded = attention_mask.expand(token_embeddings.size()).float()
                    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    q_features = sum_embeddings / sum_mask
                    q_features = F.normalize(q_features, p=2, dim=1)
                    q_embed_np = q_features.cpu().numpy().astype('float32')
            
        # Search FAISS index or compute late-interaction scores
        if dense_mode == "late-interaction":
            scores_dense_list = []
            eval_batch_size = 512 # Compute MaxSim in small batches to preserve VRAM
            for b_start in range(0, len(corpus_embeds), eval_batch_size):
                b_end = min(b_start + eval_batch_size, len(corpus_embeds))
                c_embeds_batch = corpus_embeds[b_start:b_end].to(device) # Send mini-batch of docs to GPU
                c_masks_batch = corpus_masks[b_start:b_end].to(device)
                
                sim = torch.einsum("qd,nld->nql", q_embeds, c_embeds_batch)
                sim = sim * c_masks_batch.unsqueeze(1) + (1.0 - c_masks_batch.unsqueeze(1)) * -1e9
                max_sim = torch.max(sim, dim=-1)[0]
                scores_dense_batch = torch.sum(max_sim, dim=-1)
                scores_dense_list.append(scores_dense_batch)
                
            scores_dense = torch.cat(scores_dense_list, dim=0)
        else:
            D, I = faiss_index.search(q_embed_np, len(corpus_ids))
            scores_dense = torch.zeros(len(corpus_ids), device=device)
            for rank_idx, doc_idx in enumerate(I[0]):
                scores_dense[doc_idx] = float(D[0][rank_idx])
            
        # --- 3. Score Fusion ---
        norm_sparse = min_max_normalize(scores_sparse)
        norm_dense = min_max_normalize(scores_dense)
        scores_hybrid = (1.0 - beta) * norm_sparse + beta * norm_dense
        
        # Rank document candidates
        ranked_indices = torch.argsort(scores_hybrid, descending=True).tolist()
        ranked_doc_ids = [corpus_ids[idx] for idx in ranked_indices]
        
        # Calculate MRR@10
        mrr_val = 0.0
        for rank_idx, doc_id in enumerate(ranked_doc_ids[:10]):
            if doc_id in gold_ids:
                mrr_val = 1.0 / (rank_idx + 1)
                break
        mrr_total += mrr_val
        
        # Calculate NDCG@10
        dcg = 0.0
        for rank_idx, doc_id in enumerate(ranked_doc_ids[:10]):
            if doc_id in gold_ids:
                dcg += 1.0 / (torch.log2(torch.tensor(rank_idx + 2.0)).item())
                
        # Ideal DCG
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
    print(f"Results for split '{split_name}': MRR@10 = {mrr:.4f}, NDCG@10 = {ndcg:.4f}")
    return mrr, ndcg

def main():
    parser = argparse.ArgumentParser(description="HIVE-to-V-SPLADE Hybrid Evaluation: Student (LUT-Only) + Dense Model + FAISS")
    parser.add_argument("--checkpoint-path", type=str, default="results/vsplade_student_checkpoint.pt", help="Path to distilled student weights")
    parser.add_argument("--model", type=str, default="qwen-1.5b", help="Model key of the student backbone")
    parser.add_argument("--dense-model", type=str, default="openai/clip-vit-base-patch32", help="Pretrained dense model key (e.g. openai/clip-vit-base-patch32 or sentence-transformers/all-MiniLM-L6-v2)")
    parser.add_argument("--dense-mode", type=str, default="bi-encoder", choices=["bi-encoder", "late-interaction"], help="Embedding matching architecture mode")
    parser.add_argument("--dataset", type=str, default="mm-bright/MM-BRIGHT", help="Dataset name on Hugging Face")
    parser.add_argument("--test-splits", type=str, default="academia,biology,physics,philosophy,psychology,quant,quantumcomputing,robotics,law", help="Comma-separated list of held-out splits to evaluate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for V-SPLADE document encoding")
    parser.add_argument("--dense-batch-size", type=int, default=256, help="Batch size for Dense model document encoding")
    parser.add_argument("--logit-threshold", type=float, default=0.0, help="Logit threshold to enforce sparsity")
    parser.add_argument("--filter-stopwords", action="store_true", help="Clean queries at string level by removing stop-words before BPE tokenization")
    parser.add_argument("--beta", type=float, default=0.4, help="Weight balance for dense channel scores (score_hybrid = (1-beta)*sparse + beta*dense)")
    args = parser.parse_args()
    
    device = best_gpu()
    
    # 1. Load V-SPLADE Student
    print(f"Loading base student model '{args.model}' on {device}...")
    model, tokenizer = load(args.model, device=device)
    if hasattr(model, "device"):
        device = model.device
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
    print("Models loaded successfully.")
    
    # Parse test splits
    test_splits = [s.strip() for s in args.test_splits.split(",") if s.strip()]
    print(f"Evaluating splits: {test_splits}")
    
    all_results = {}
    mrr_sum = 0.0
    ndcg_sum = 0.0
    evaluated_count = 0
    
    for split in test_splits:
        res = evaluate_split_hybrid(
            model, tokenizer, query_lut, dense_model, dense_processor, is_clip, device, 
            args.dataset, split, batch_size=args.batch_size, dense_batch_size=args.dense_batch_size, 
            logit_threshold=args.logit_threshold, filter_stopwords=args.filter_stopwords, beta=args.beta,
            dense_mode=args.dense_mode
        )
        if res:
            mrr, ndcg = res
            all_results[split] = {"MRR@10": mrr, "NDCG@10": ndcg}
            mrr_sum += mrr
            ndcg_sum += ndcg
            evaluated_count += 1
            
    if evaluated_count > 0:
        print("\n==========================================")
        print("Summary of Zero-Shot Hybrid Cross-Domain Evaluation:")
        print("==========================================\n")
        for split, metrics in all_results.items():
            print(f"Split: {split:<20} | MRR@10 = {metrics['MRR@10']:.4f} | NDCG@10 = {metrics['NDCG@10']:.4f}")
        print("-" * 50)
        print(f"AVERAGE (OOD Held-Out): MRR@10 = {mrr_sum/evaluated_count:.4f} | NDCG@10 = {ndcg_sum/evaluated_count:.4f}")
        print("==========================================\n")
        
        # Save evaluation summary
        output_results_path = "results/hybrid_evaluation_results.json"
        with open(output_results_path, "w") as f:
            json.dump({
                "dense_model": dense_model_name,
                "individual_splits": all_results,
                "average": {"MRR@10": mrr_sum/evaluated_count, "NDCG@10": ndcg_sum/evaluated_count}
            }, f, indent=2)
        print(f"Results saved to {output_results_path}")

if __name__ == "__main__":
    main()
