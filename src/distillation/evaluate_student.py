import os
import sys
import json
import argparse
import random
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset

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
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Keep only alphanumeric characters and spaces
    text = ''.join(c for c in text if c.isalnum() or c.isspace())
    # Filter stop words
    words = text.split()
    filtered = [w for w in words if w.lower() not in stop_words]
    return ' '.join(filtered)

def evaluate_split(model, tokenizer, query_lut, device, dataset_name, split_name, batch_size=32, logit_threshold=0.0, filter_stopwords=False):
    """Evaluate V-SPLADE student model on a single split, returning MRR@10 and NDCG@10."""
    print(f"\nEvaluating split: '{split_name}'...")
    
    # Load dataset split docs & examples
    try:
        docs_ds = load_dataset(dataset_name, "documents", split=split_name)
        queries_ds = load_dataset(dataset_name, "examples", split=split_name)
    except Exception as e:
        print(f"Error loading split '{split_name}': {e}")
        return None
        
    print(f"Loaded {len(docs_ds)} documents and {len(queries_ds)} queries.")
    
    # Pre-encode all documents in batches to save time
    corpus_ids = []
    w_p_list = []
    
    model.eval()
    vocab_size = model.config.vocab_size
    
    print("Encoding visual documents...")
    with torch.no_grad():
        for i in tqdm(range(0, len(docs_ds), batch_size), desc="Doc Batches"):
            batch = docs_ds[i : i + batch_size]
            
            # Since slicing a Hugging Face Dataset returns a dictionary of lists, extract columns directly:
            doc_texts = batch.get("content", batch.get("text", batch.get("caption", batch.get("llm_image_caption", []))))
            if not doc_texts:
                doc_texts = ["[No text extraction available]"] * len(batch.get("id", []))
            
            # Clean text values (replace None with empty strings)
            doc_texts = [text if text is not None else "[No text]" for text in doc_texts]
            
            doc_ids = batch.get("id", [])
            corpus_ids.extend(doc_ids)
            
            doc_inputs = tokenizer(
                doc_texts,
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
            hidden_states = outputs.hidden_states[-1] # (batch, seq, hidden)
            
            # Max-pool logits in chunks to prevent VRAM spikes
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
            
            # Apply logit thresholding
            if logit_threshold > 0.0:
                z_p = z_p - logit_threshold
                
            w_p = torch.log1p(torch.relu(z_p)) # (batch, vocab_size)
            w_p_list.append(w_p.cpu())
            
    # Concatenate all doc sparse representations
    w_p_all = torch.cat(w_p_list, dim=0) # (num_docs, vocab_size)
    
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
            
        # Clean query text at the string level if requested (preventing BPE subword degradation)
        if filter_stopwords:
            q_text_cleaned = clean_query_text(q_text)
        else:
            q_text_cleaned = q_text
            
        # Get query tokens and weights via LUT
        query_token_ids = tokenizer(q_text_cleaned, add_special_tokens=False)["input_ids"]
        if not query_token_ids:
            continue
            
        w_q = torch.zeros(vocab_size, dtype=torch.float32)
        # Load weights from LUT
        with torch.no_grad():
            w_q_weights = F.softplus(query_lut[query_token_ids]).cpu()
            w_q[query_token_ids] = w_q_weights
            
        # Match via dot-product: s = w_q^T w_p
        scores = torch.sum(w_q.unsqueeze(0) * w_p_all, dim=1) # (num_docs,)
        
        # Rank document candidates
        ranked_indices = torch.argsort(scores, descending=True).tolist()
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
    parser = argparse.ArgumentParser(description="HIVE-to-V-SPLADE Phase 3: Student Evaluation (LUT-Only)")
    parser.add_argument("--checkpoint-path", type=str, default="results/vsplade_student_checkpoint.pt", help="Path to distilled student weights")
    parser.add_argument("--model", type=str, default="qwen-1.5b", help="Model key of the backbone")
    parser.add_argument("--dataset", type=str, default="mm-bright/MM-BRIGHT", help="Dataset name on Hugging Face")
    parser.add_argument("--test-splits", type=str, default="academia,biology,physics,philosophy,psychology,quant,quantumcomputing,robotics,law", help="Comma-separated list of held-out splits to evaluate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for encoding documents")
    parser.add_argument("--logit-threshold", type=float, default=0.0, help="Logit threshold to enforce sparsity")
    parser.add_argument("--filter-stopwords", action="store_true", help="Clean queries at string level by removing stop-words before BPE tokenization")
    args = parser.parse_args()
    
    device = best_gpu()
    print(f"Loading base model '{args.model}' on {device}...")
    model, tokenizer = load(args.model, device=device)
    
    # Load checkpoint
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint file not found at {args.checkpoint_path}.")
        return
        
    print(f"Loading checkpoint weights from {args.checkpoint_path}...")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    query_lut = checkpoint["query_lut"].to(device)
    print("Checkpoint loaded successfully.")
    
    # Parse test splits
    test_splits = [s.strip() for s in args.test_splits.split(",") if s.strip()]
    print(f"Evaluating splits: {test_splits}")
    
    all_results = {}
    mrr_sum = 0.0
    ndcg_sum = 0.0
    evaluated_count = 0
    
    for split in test_splits:
        res = evaluate_split(
            model, tokenizer, query_lut, device, args.dataset, split, 
            batch_size=args.batch_size, logit_threshold=args.logit_threshold, filter_stopwords=args.filter_stopwords
        )
        if res:
            mrr, ndcg = res
            all_results[split] = {"MRR@10": mrr, "NDCG@10": ndcg}
            mrr_sum += mrr
            ndcg_sum += ndcg
            evaluated_count += 1
            
    if evaluated_count > 0:
        print("\n==========================================")
        print("Summary of Zero-Shot Cross-Domain Evaluation:")
        print("==========================================\n")
        for split, metrics in all_results.items():
            print(f"Split: {split:<20} | MRR@10 = {metrics['MRR@10']:.4f} | NDCG@10 = {metrics['NDCG@10']:.4f}")
        print("-" * 50)
        print(f"AVERAGE (OOD Held-Out): MRR@10 = {mrr_sum/evaluated_count:.4f} | NDCG@10 = {ndcg_sum/evaluated_count:.4f}")
        print("==========================================\n")
        
        # Save evaluation summary
        output_results_path = "results/student_evaluation_results.json"
        with open(output_results_path, "w") as f:
            json.dump({
                "individual_splits": all_results,
                "average": {"MRR@10": mrr_sum/evaluated_count, "NDCG@10": ndcg_sum/evaluated_count}
            }, f, indent=2)
        print(f"Results saved to {output_results_path}")

if __name__ == "__main__":
    main()
