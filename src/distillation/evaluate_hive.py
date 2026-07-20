import os
import sys
import json
import argparse
import numpy as np

# Add project root parent to sys.path to find src
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

def evaluate_hive_targets(targets_path):
    if not os.path.exists(targets_path):
        print(f"Error: Targets file not found at {targets_path}")
        return
        
    print(f"Loading HIVE teacher targets from {targets_path}...")
    with open(targets_path, "r") as f:
        targets = json.load(f)
        
    print(f"Loaded {len(targets)} query targets.")
    
    # Group queries by split
    splits_data = {}
    for entry in targets:
        split = entry.get("split", "unknown")
        splits_data.setdefault(split, []).append(entry)
        
    print(f"Detected splits: {list(splits_data.keys())}")
    
    all_results = {}
    overall_mrr = []
    overall_ndcg = []
    
    for split_name, entries in splits_data.items():
        mrr_list = []
        ndcg_list = []
        
        for entry in entries:
            gold_ids = entry.get("ground_truth_ids", [])
            candidate_ids = entry.get("candidate_ids", [])
            scores = entry.get("relevance_scores", {})
            
            if not gold_ids:
                continue
                
            # Sort candidates by relevance score (descending)
            # Default to 0.0 if a candidate has no score
            scored_candidates = []
            for doc_id in candidate_ids:
                score_val = scores.get(doc_id, 0.0)
                # Parse to float if it is a string
                try:
                    score_val = float(score_val)
                except ValueError:
                    score_val = 0.0
                scored_candidates.append((doc_id, score_val))
                
            # Sort: highest score first. Break ties stably.
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            sorted_doc_ids = [item[0] for item in scored_candidates]
            
            # Calculate MRR@10
            mrr_val = 0.0
            for rank_idx, doc_id in enumerate(sorted_doc_ids[:10]):
                if doc_id in gold_ids:
                    mrr_val = 1.0 / (rank_idx + 1)
                    break
            mrr_list.append(mrr_val)
            overall_mrr.append(mrr_val)
            
            # Calculate NDCG@10
            dcg = 0.0
            for rank_idx, doc_id in enumerate(sorted_doc_ids[:10]):
                if doc_id in gold_ids:
                    dcg += 1.0 / (np.log2(rank_idx + 2))
                    
            idcg = 0.0
            for rank_idx in range(min(10, len(gold_ids))):
                idcg += 1.0 / (np.log2(rank_idx + 2))
                
            ndcg_val = dcg / idcg if idcg > 0.0 else 0.0
            ndcg_list.append(ndcg_val)
            overall_ndcg.append(ndcg_val)
            
        if len(mrr_list) > 0:
            avg_mrr = np.mean(mrr_list)
            avg_ndcg = np.mean(ndcg_list)
            all_results[split_name] = {
                "MRR@10": avg_mrr,
                "NDCG@10": avg_ndcg,
                "queries_count": len(mrr_list)
            }
            
    # Print results
    print("\n==========================================")
    print("Summary of HIVE Teacher Evaluation:")
    print("==========================================\n")
    for split, metrics in all_results.items():
        print(f"Split: {split:<20} ({metrics['queries_count']} queries) | MRR@10 = {metrics['MRR@10']:.4f} | NDCG@10 = {metrics['NDCG@10']:.4f}")
    print("-" * 50)
    if len(overall_mrr) > 0:
        print(f"AVERAGE (Macro):          | MRR@10 = {np.mean(overall_mrr):.4f} | NDCG@10 = {np.mean(overall_ndcg):.4f}")
    print("==========================================\n")

def main():
    parser = argparse.ArgumentParser(description="Evaluate HIVE Teacher Retrieval Metrics from Targets Cache")
    parser.add_argument("--targets-path", type=str, default="results/hive_test_targets.json", help="Path to HIVE targets cache JSON file")
    args = parser.parse_args()
    
    evaluate_hive_targets(args.targets_path)

if __name__ == "__main__":
    main()
