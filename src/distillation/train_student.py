import os
import sys
import json
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset

# Add project root parent to sys.path to find src
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.models import load, best_gpu

def load_all_documents(targets):
    """Load all document texts dynamically based on the dataset and split in each target."""
    corpus = {}
    # Group splits by dataset
    grouped = {}
    for t in targets:
        dataset = t.get("dataset", "mm-bright/MM-BRIGHT")
        split = t["split"]
        grouped.setdefault(dataset, set()).add(split)
        
    for dataset, splits in grouped.items():
        for split in splits:
            print(f"Loading documents for dataset '{dataset}', split '{split}'...")
            try:
                if "mm-bright" in dataset.lower():
                    ds = load_dataset(dataset, "documents", split=split)
                    for row in ds:
                        doc_id = row.get("id")
                        doc_text = row.get("content", row.get("text", row.get("caption", row.get("llm_image_caption", ""))))
                        if not doc_text:
                            doc_text = "[No text extraction available]"
                        corpus[doc_id] = doc_text
                elif dataset.startswith("beir/") or "beir" in dataset.lower():
                    # Load BEIR corpus configuration directly
                    print(f"Detected BEIR dataset configuration for '{dataset}' corpus...")
                    ds = load_dataset(dataset, "corpus", split="corpus")
                    for row in ds:
                        raw_doc_id = row["_id"]
                        # Match the exact namespaced document IDs used in generate_teacher_data.py
                        doc_id = f"{dataset.replace('/', '_')}_{raw_doc_id}"
                        corpus[doc_id] = row.get("text", "")
                else:
                    ds = load_dataset(dataset, split=split)
                    for idx, row in enumerate(ds):
                        doc_id = row.get("doc_id") or row.get("image_filename") or f"{split}_DOC-{idx}"
                        doc_text = (row.get("text") or row.get("caption") or row.get("description") or "").strip()
                        if not doc_text:
                            doc_text = (row.get("answer") or row.get("page") or "[No text content]").strip()
                        corpus[doc_id] = doc_text
            except Exception as e:
                print(f"Error loading documents for dataset '{dataset}' split '{split}': {e}")
    return corpus

def main():
    parser = argparse.ArgumentParser(description="HIVE-to-V-SPLADE Phase 2: Student Distillation (LUT-Only)")
    parser.add_argument("--targets-path", type=str, default="results/mm_bright_train_teacher_targets.json", help="Path to HIVE generated teacher targets")
    parser.add_argument("--model", type=str, default="qwen-1.5b", help="Model key to load from src/models.py as the backbone")
    parser.add_argument("--output-path", type=str, default="results/vsplade_student_checkpoint.pt", help="Path to save the distilled student weights")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr-query-lut", type=float, default=1e-3, help="Learning rate for query LUT")
    parser.add_argument("--lambda-rank", type=float, default=1.0, help="Weight for Rank Alignment loss")
    parser.add_argument("--lambda-sparsity", type=float, default=1e-4, help="Weight for Sparsity regularization")
    parser.add_argument("--tau-cap", type=float, default=1.0, help="Temperature for caption/gating sharpening")
    parser.add_argument("--dry-run", action="store_true", help="Run a quick dry run on 2 samples")
    args = parser.parse_args()
    
    device = best_gpu()
    print(f"Loading student base model '{args.model}' on {device}...")
    model, tokenizer = load(args.model, device=device)
    
    # Load targets dataset
    if not os.path.exists(args.targets_path):
        print(f"Error: Targets file not found at {args.targets_path}. Please run Phase 1 first.")
        return
        
    with open(args.targets_path, "r") as f:
        targets = json.load(f)
    print(f"Loaded {len(targets)} teacher targets for training.")
    
    if args.dry_run:
        targets = targets[:2]
        print(f"Dry run: limited targets to {len(targets)} samples.")
    
    # Find all unique splits to load corpus documents
    print("Loading document corpora from targets...")
    corpus = load_all_documents(targets)
    print(f"Total corpus documents loaded: {len(corpus)}")
    
    # Freeze model completely (backbone and lm_head)
    for p in model.parameters():
        p.requires_grad = False
    
    # Initialize query lookup table (Lut) as a trainable 1D tensor parameter
    vocab_size = model.config.vocab_size
    query_lut = nn.Parameter(torch.ones(vocab_size, dtype=torch.float32, device=device))
    
    # Optimize ONLY query_lut
    optimizer = torch.optim.AdamW([
        {"params": [query_lut], "lr": args.lr_query_lut}
    ], weight_decay=0.01)
    
    print("Starting distillation training loop (LUT-Only)...")
    for epoch in range(args.epochs):
        random.seed(epoch)
        random.shuffle(targets)
        
        epoch_loss = 0.0
        epoch_lexical = 0.0
        epoch_rank = 0.0
        epoch_sparsity = 0.0
        
        progress_bar = tqdm(targets, desc=f"Epoch {epoch+1}/{args.epochs}")
        for step, target in enumerate(progress_bar):
            q_text = target["query_text"]
            comp_query = target["compensatory_query"]
            candidate_ids = target["candidate_ids"]
            rationales = target["rationales"]
            relevance_scores = target["relevance_scores"]
            
            # 1. Prepare Document Batch inputs
            doc_texts = [corpus.get(d_id, "[Document Content Missing]") for d_id in candidate_ids]
            
            # Sub-batch documents to prevent CUDA Out-of-Memory (OOM) spikes
            doc_sub_batch_size = 4
            w_p_parts = []
            
            for b_idx in range(0, len(doc_texts), doc_sub_batch_size):
                sub_doc_texts = doc_texts[b_idx : b_idx + doc_sub_batch_size]
                
                # Tokenize documents
                doc_inputs = tokenizer(
                    sub_doc_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(device)
                
                # 2. Forward pass through frozen model
                with torch.no_grad():
                    outputs = model.model(
                        input_ids=doc_inputs["input_ids"],
                        attention_mask=doc_inputs["attention_mask"],
                        output_hidden_states=True
                    )
                    hidden_states = outputs.hidden_states[-1] # (batch_size, seq_len, hidden_dim)
                    
                    attention_mask = doc_inputs["attention_mask"].unsqueeze(-1) # (batch_size, seq_len, 1)
                    
                    # Compute logits in chunks
                    chunk_size = 20000
                    z_p_parts = []
                    for i in range(0, vocab_size, chunk_size):
                        weight_chunk = model.lm_head.weight[i:i+chunk_size]
                        logits_chunk = torch.matmul(hidden_states, weight_chunk.t())
                        logits_chunk = logits_chunk * attention_mask + (1 - attention_mask) * -1e9
                        z_p_chunk, _ = torch.max(logits_chunk, dim=1)
                        z_p_parts.append(z_p_chunk)
                        
                    z_p = torch.cat(z_p_parts, dim=1) # (batch_size, vocab_size)
                    sub_w_p = torch.log1p(torch.relu(z_p)) # (batch_size, vocab_size)
                    w_p_parts.append(sub_w_p)
                    
            w_p = torch.cat(w_p_parts, dim=0) # (candidate_pool_size, vocab_size)
            batch_size = w_p.shape[0]
            
            # 3. Process Query inputs and representations
            query_token_ids = tokenizer(q_text, add_special_tokens=False)["input_ids"]
            if not query_token_ids:
                continue
                
            # Construct student query vector: w_q[v] = Softplus(query_lut[v]) for query tokens
            w_q = torch.zeros(vocab_size, dtype=w_p.dtype, device=device)
            w_q[query_token_ids] = F.softplus(query_lut[query_token_ids])
            
            # Compute student scores: s_SPLADE = w_q^T w_p
            scores_student = torch.sum(w_q.unsqueeze(0) * w_p, dim=1) # (batch_size,)
            
            # 4. Compute Lexical Gating Loss (Objective 1)
            loss_lexical = 0.0
            q_tokens = set(query_token_ids)
            comp_tokens = set(tokenizer(comp_query, add_special_tokens=False)["input_ids"])
            
            for idx, doc_id in enumerate(candidate_ids):
                rat = rationales.get(doc_id, "")
                rat_tokens = set(tokenizer(rat, add_special_tokens=False)["input_ids"])
                
                union_tokens = list(q_tokens.union(comp_tokens).union(rat_tokens))
                
                W_R = torch.zeros(vocab_size, dtype=w_p.dtype, device=device)
                if union_tokens:
                    W_R[union_tokens] = 1.0
                    
                w_p_sg = w_p[idx].detach()
                o_reason = w_p_sg * W_R
                
                o_temp = o_reason ** (1.0 / args.tau_cap)
                sum_o = o_temp.sum()
                if sum_o > 0:
                    alpha = o_temp / sum_o
                else:
                    alpha = torch.zeros_like(o_reason)
                    if union_tokens:
                        alpha[union_tokens] = 1.0 / len(union_tokens)
                        
                loss_lexical += -torch.sum(alpha * F.logsigmoid(z_p[idx]))
                
            loss_lexical = loss_lexical / batch_size
            
            # 5. Compute Rank Alignment Loss (Objective 2: Margin MSE)
            scores_teacher = [float(relevance_scores.get(d_id, 0.0)) for d_id in candidate_ids]
            
            margins_student = []
            margins_teacher = []
            for i in range(batch_size):
                for j in range(i + 1, batch_size):
                    margins_student.append(scores_student[i] - scores_student[j])
                    margins_teacher.append(scores_teacher[i] - scores_teacher[j])
                    
            if margins_student:
                margins_student = torch.stack(margins_student)
                margins_teacher = torch.tensor(margins_teacher, dtype=margins_student.dtype, device=device)
                loss_rank = F.mse_loss(margins_student, margins_teacher)
            else:
                loss_rank = torch.tensor(0.0, device=device)
                
            # 6. Compute Sparsity Regularization
            loss_sparsity = torch.mean(w_p)
            
            # Joint Objective
            loss_total = loss_lexical + args.lambda_rank * loss_rank + args.lambda_sparsity * loss_sparsity
            
            # Backward pass & Optimize
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            
            # Track statistics
            epoch_loss += loss_total.item()
            epoch_lexical += loss_lexical.item()
            epoch_rank += loss_rank.item()
            epoch_sparsity += loss_sparsity.item()
            
            if step % 20 == 0:
                progress_bar.set_postfix({
                    "loss": f"{loss_total.item():.4f}",
                    "lexical": f"{loss_lexical.item():.4f}",
                    "rank": f"{loss_rank.item():.4f}",
                    "sparse": f"{loss_sparsity.item():.5f}"
                })
                
        # Epoch summary
        n_steps = len(targets)
        print(f"\nEpoch {epoch+1} Complete:")
        print(f"  Avg Loss: {epoch_loss/n_steps:.4f}")
        print(f"  Avg Lexical Loss: {epoch_lexical/n_steps:.4f}")
        print(f"  Avg Rank Loss: {epoch_rank/n_steps:.4f}")
        print(f"  Avg Sparsity Loss: {epoch_sparsity/n_steps:.5f}\n")
        
    # Save checkpoint (LUT-only)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    print(f"Saving student query LUT weight dictionary to {args.output_path}...")
    torch.save({
        "query_lut": query_lut.detach().cpu(),
        "args": vars(args)
    }, args.output_path)
    print("Distillation Training Complete.")

if __name__ == "__main__":
    main()
