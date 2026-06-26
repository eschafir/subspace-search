import os
import sys
import json
import argparse
import random
import torch
import collections
import re
from rank_bm25 import BM25Okapi

# Add project root parent to sys.path to find src
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()  # load environment variables from .env file

from src.models import load, best_gpu

def run_llm_inference(prompt: str, model, tokenizer, device: str, max_new_tokens: int = 128) -> str:
    """Helper to run inference on local Qwen or custom HF model, or Nvidia NIM API."""
    if isinstance(model, str) and model.startswith("nvidia/"):
        # Nvidia NIM API call
        from openai import OpenAI
        api_key = os.environ.get("NVIDIA_API_KEY")
        if not api_key:
            raise ValueError("Error: NVIDIA_API_KEY is not set in environment or .env file.")
            
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )
        
        # Strip the 'nvidia/' prefix to get the official NIM model key
        nim_model_name = model.replace("nvidia/", "")
        
        response = client.chat.completions.create(
            model=nim_model_name,
            messages=[
                {"role": "system", "content": "You are a helpful, logical AI search assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=max_new_tokens
        )
        return response.choices[0].message.content.strip()
        
    # Local HF model fallback
    messages = [
        {"role": "system", "content": "You are a helpful, logical AI search assistant."},
        {"role": "user", "content": prompt}
    ]
    
    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
        
    response = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return response

def parse_stage4_response(response: str) -> tuple[str, float]:
    """Parse step-by-step rationale and numeric score from Stage 4 LLM output."""
    # Look for a score at the end of the text like Score: 4 or [Score]: 4.5
    score_match = re.search(r"(?:score|relevance score|rating)[:\s]+([0-5](?:\.\d+)?)", response, re.IGNORECASE)
    score = 0.0
    if score_match:
        score = float(score_match.group(1))
        
    # Extract rationale (everything preceding the score declaration or the full text)
    rationale = response
    score_decl_match = re.search(r"(?:score|relevance score|rating)[:\s]+", response, re.IGNORECASE)
    if score_decl_match:
        rationale = response[:score_decl_match.start()].strip()
        
    return rationale, score

def main():
    parser = argparse.ArgumentParser(description="HIVE Phase 1: Offline Teacher Target Generation")
    parser.add_argument("--model", type=str, default="qwen-1.5b", help="Model key to load from src/models.py or NIM API path")
    parser.add_argument("--dataset", type=str, default="nfcorpus", help="Dataset name: 'nfcorpus', 'mm-bright/MM-BRIGHT', or a Hugging Face dataset ID")
    parser.add_argument("--split", type=str, default=None, help="Hugging Face dataset split/domain name (e.g. 'academia', or 'all' for all MM-BRIGHT splits)")
    parser.add_argument("--num-queries", type=int, default=10, help="Number of queries to process per dataset split")
    parser.add_argument("--haystack-size", type=int, default=10, help="Candidate pool size for Stage 4 reranking")
    parser.add_argument("--output-path", type=str, default="results/distillation_targets.json", help="Output path for the generated targets")
    args = parser.parse_args()
    
    if args.model.startswith("nvidia/"):
        print(f"Using Nvidia NIM API model: {args.model}")
        model = args.model
        tokenizer = None
        device = "cpu"
    else:
        device = best_gpu()
        print(f"Loading teacher model {args.model} on {device}...")
        model, tokenizer = load(args.model, device=device)
    
    # Load existing targets if file exists to enable resuming
    targets = []
    processed_q_ids = set()
    if os.path.exists(args.output_path):
        print(f"Loading existing distillation targets from {args.output_path} to resume...")
        try:
            with open(args.output_path, "r") as f:
                targets = json.load(f)
            processed_q_ids = {t["query_id"] for t in targets}
            print(f"Loaded {len(targets)} already processed query targets.")
        except Exception as e:
            print(f"Warning: Failed to load existing targets file: {e}. Starting fresh.")
            
    # Determine the split list to iterate over
    if args.dataset != "nfcorpus" and ("/" in args.dataset or args.dataset.startswith("vidore/")):
        try:
            from datasets import load_dataset, get_dataset_split_names
        except ImportError:
            print("Error: Hugging Face 'datasets' library is not installed in this environment.")
            return
            
        if "mm-bright" in args.dataset.lower():
            if args.split == "all":
                try:
                    split_names = get_dataset_split_names(args.dataset, "examples")
                    print(f"Detected {len(split_names)} splits in MM-BRIGHT to process: {split_names}")
                except Exception as e:
                    print(f"Error listing MM-BRIGHT splits: {e}")
                    return
            elif args.split and "," in args.split:
                split_names = [s.strip() for s in args.split.split(",") if s.strip()]
                print(f"User specified {len(split_names)} splits to process: {split_names}")
            else:
                split_names = [args.split if args.split else "academia"]
        else:
            split_names = [args.split if args.split else "test"]
    else:
        split_names = ["local"] # local NFCorpus
        
    # Process each split in a loop
    for split_idx, split_name in enumerate(split_names):
        print(f"\n==========================================")
        print(f"Processing split {split_idx+1}/{len(split_names)}: {split_name}")
        print(f"==========================================\n")
        
        corpus = {}
        queries = {}
        qrels = {}
        
        # Load the split data
        if args.dataset != "nfcorpus" and ("/" in args.dataset or args.dataset.startswith("vidore/")):
            if "mm-bright" in args.dataset.lower():
                print(f"Loading MM-BRIGHT dataset under configuration 'documents' and 'examples', split: '{split_name}'...")
                try:
                    docs_ds = load_dataset(args.dataset, "documents", split=split_name)
                    queries_ds = load_dataset(args.dataset, "examples", split=split_name)
                except Exception as e:
                    print(f"Error loading MM-BRIGHT split '{split_name}' from Hugging Face: {e}")
                    continue
                    
                print(f"Successfully loaded MM-BRIGHT {split_name} split: {len(docs_ds)} documents, {len(queries_ds)} queries.")
                
                # Map documents
                for row in docs_ds:
                    doc_id = row.get("id")
                    doc_text = row.get("content", row.get("text", row.get("caption", row.get("llm_image_caption", ""))))
                    if not doc_text:
                        doc_text = "[No text extraction available]"
                    corpus[doc_id] = {"_id": doc_id, "title": f"Document {doc_id}", "text": doc_text}
                    
                # Map queries and relevance
                for row in queries_ds:
                    q_id = row.get("id")
                    query_text = row.get("query", "")
                    queries[q_id] = {"_id": q_id, "text": query_text}
                    qrels[q_id] = row.get("gold_ids", [])
            else:
                # Generic loader for ViDoRe or other HF datasets
                print(f"Loading HF dataset split: '{split_name}'...")
                try:
                    ds = load_dataset(args.dataset, split=split_name)
                except Exception as e:
                    print(f"Error loading Hugging Face dataset split '{split_name}': {e}")
                    continue
                    
                print(f"Successfully loaded HF dataset split ({len(ds)} rows).")
                
                # Map Hugging Face fields to standard corpus/queries format
                for idx, row in enumerate(ds):
                    q_id = f"{split_name}_Q-{idx}"
                    query_text = (row.get("query") or "").strip()
                    if not query_text:
                        continue
                        
                    queries[q_id] = {"_id": q_id, "text": query_text}
                    
                    doc_id = row.get("doc_id") or row.get("image_filename") or f"{split_name}_DOC-{idx}"
                    doc_title = f"Document: {doc_id}"
                    
                    # Try answer/page if text/caption are missing (common in synthetic datasets)
                    doc_text = (row.get("text") or row.get("caption") or row.get("description") or "").strip()
                    if not doc_text:
                        doc_text = (row.get("answer") or row.get("page") or "[No text content]").strip()
                        
                    corpus[doc_id] = {"_id": doc_id, "title": doc_title, "text": doc_text}
                    qrels.setdefault(q_id, []).append(doc_id)
        else:
            # Load local NFCorpus
            print("Loading local NFCorpus dataset...")
            corpus_path = os.path.join(project_root, "data", "nfcorpus", "corpus.jsonl")
            if not os.path.exists(corpus_path):
                print(f"Error: Corpus not found at {corpus_path}. Please download NFCorpus first.")
                return
                
            with open(corpus_path, "r") as f:
                for line in f:
                    doc = json.loads(line)
                    corpus[doc["_id"]] = doc
                    
            queries_path = os.path.join(project_root, "data", "nfcorpus", "queries.jsonl")
            with open(queries_path, "r") as f:
                for line in f:
                    q = json.loads(line)
                    queries[q["_id"]] = q
                    
            qrels_path = os.path.join(project_root, "data", "nfcorpus", "qrels", "test.tsv")
            with open(qrels_path, "r") as f:
                f.readline()  # header
                for line in f:
                    q_id, doc_id, score = line.strip().split("\t")
                    if int(score) >= 2:
                        qrels.setdefault(q_id, []).append(doc_id)
                        
        # Filter queries with valid relevance judgments and not already processed
        valid_q_ids = [q_id for q_id in queries.keys() if q_id in qrels and q_id not in processed_q_ids]
        if not valid_q_ids:
            print(f"All valid queries in split '{split_name}' have already been processed. Skipping split.")
            continue
            
        random.seed(42)
        selected_q_ids = random.sample(valid_q_ids, min(args.num_queries, len(valid_q_ids)))
        print(f"Selected {len(selected_q_ids)} new queries to process for split '{split_name}' (out of {len(valid_q_ids)} remaining).")
        
        # Pre-tokenize corpus for BM25
        corpus_ids = list(corpus.keys())
        corpus_texts = [f"{corpus[d_id]['title']} {corpus[d_id]['text']}" for d_id in corpus_ids]
        corpus_tokens = [[w.strip() for w in re.split(r"\W+", text.lower()) if w.strip()] for text in corpus_texts]
        bm25 = BM25Okapi(corpus_tokens)
        
        for idx, q_id in enumerate(selected_q_ids):
            query_text = queries[q_id]["text"]
            print(f"\n--- Processing Query {idx+1}/{len(selected_q_ids)} (Split: {split_name}): '{query_text}' ---")
            
            # 1. HIVE Stage 1: Initial Retrieval
            query_tokens = [w.strip() for w in re.split(r"\W+", query_text.lower()) if w.strip()]
            initial_scores = bm25.get_scores(query_tokens)
            initial_ranked_indices = sorted(range(len(corpus_ids)), key=lambda i: initial_scores[i], reverse=True)
            top_k1_ids = [corpus_ids[i] for i in initial_ranked_indices[:5]]
            
            # Format top-k1 snippets for Stage 2 Query Synthesis
            snippets = []
            for d_id in top_k1_ids:
                doc = corpus[d_id]
                snippets.append(f"Document ID: {d_id}\nTitle: {doc['title']}\nExcerpt: {doc['text'][:200]}...")
            snippets_text = "\n\n".join(snippets)
            
            # 2. HIVE Stage 2: Compensatory Query Synthesis
            stage2_prompt = (
                f"Original Search Query: \"{query_text}\"\n\n"
                f"Here are the top retrieved document snippets from a first-pass search:\n"
                f"{snippets_text}\n\n"
                f"Determine the logical/semantic reasoning gap or missing vocabulary concepts "
                f"between the original query and the retrieved documents. "
                f"Synthesize a single 'compensatory' search query containing synonyms, expanded medical terms, "
                f"or logical concepts to retrieve other relevant documents that bridge this gap. "
                f"Output ONLY the synthesized compensatory query, with no other text."
            )
            print("Running Stage 2: Compensatory Query Synthesis...")
            comp_query = run_llm_inference(stage2_prompt, model, tokenizer, device, max_new_tokens=64)
            print(f"  Compensatory Query: '{comp_query}'")
            
            # 3. HIVE Stage 3: Secondary Retrieval
            comp_tokens = [w.strip() for w in re.split(r"\W+", comp_query.lower()) if w.strip()]
            secondary_scores = bm25.get_scores(comp_tokens)
            secondary_ranked_indices = sorted(range(len(corpus_ids)), key=lambda i: secondary_scores[i], reverse=True)
            top_k2_ids = [corpus_ids[i] for i in secondary_ranked_indices[:args.haystack_size]]
            
            # Union candidate pool (keeping ground truth needle in there if it exists)
            candidate_pool = list(set(top_k1_ids + top_k2_ids))
            ground_truth_ids = qrels[q_id]
            # Guarantee ground truth needle is in the pool to obtain positive labels
            for gt_id in ground_truth_ids:
                if gt_id not in candidate_pool:
                    candidate_pool.append(gt_id)
            
            print(f"Candidate pool size for Stage 4: {len(candidate_pool)}")
            
            # 4. HIVE Stage 4: Verification and Reranking
            rationales = {}
            scores = {}
            
            for doc_id in candidate_pool:
                doc = corpus[doc_id]
                is_ground_truth = "Yes" if doc_id in ground_truth_ids else "No"
                
                stage4_prompt = (
                    f"Original User Query: \"{query_text}\"\n\n"
                    f"Candidate Document (ID: {doc_id}):\n"
                    f"Title: {doc['title']}\n"
                    f"Content: {doc['text']}\n\n"
                    f"Analyze if this document is logically relevant to answering the original user query. "
                    f"Explain your step-by-step reasoning (the rationale). "
                    f"Then, assign a relevance score between 0.0 (completely irrelevant) and 5.0 (perfect answer). "
                    f"Output your response in the format:\n"
                    f"Rationale: <your reasoning here>\n"
                    f"Score: <number from 0.0 to 5.0>"
                )
                print(f"  Verifying Doc {doc_id} (Is Ground Truth: {is_ground_truth})...")
                verification_output = run_llm_inference(stage4_prompt, model, tokenizer, device, max_new_tokens=150)
                
                rationale, score = parse_stage4_response(verification_output)
                rationales[doc_id] = rationale
                scores[doc_id] = score
                print(f"    Assigned Score: {score}")
                
            # Compile distillation targets
            target_entry = {
                "query_id": q_id,
                "split": split_name,
                "query_text": query_text,
                "compensatory_query": comp_query,
                "ground_truth_ids": ground_truth_ids,
                "candidate_ids": candidate_pool,
                "rationales": rationales,
                "relevance_scores": scores
            }
            targets.append(target_entry)
            processed_q_ids.add(q_id)
            
            # Create output directories if needed and save progressively
            os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
            with open(args.output_path, "w") as f:
                json.dump(targets, f, indent=2)
            print(f"Saved progress. Total targets compiled so far: {len(targets)}")
            
    print(f"\nPhase 1 Complete. Distillation targets saved to {args.output_path}")

if __name__ == "__main__":
    main()
