# evaluation.py
# Retrieval evaluation for Medical RAG System
# STATUS: Framework designed, ground-truth annotation pending

import json
from typing import List, Dict, Tuple

class RetrievalEvaluator:
    """
    Evaluates retrieval quality of the Medical RAG system.
    
    NOTE: This requires a ground-truth test set where each query
    is mapped to the relevant chunk ID(s). That annotation is
    pending — this script defines the evaluation methodology.
    """
    
    def __init__(self, rag_system):
        self.rag = rag_system
        self.test_set = []  # To be populated with annotated queries
    
    def load_test_set(self, path: str):
        """
        Test set format:
        [
            {
                "query": "What are symptoms of migraine?",
                "relevant_chunk_ids": ["kg_12_12", "kg_45_45"],
                "relevant_condition": "Migraine"
            },
            ...
        ]
        """
        with open(path, 'r') as f:
            self.test_set = json.load(f)
    
    def recall_at_k(self, k: int = 5) -> float:
        """
        Recall@k: fraction of queries where at least one relevant
        chunk appears in the top-k retrieved results.
        """
        if not self.test_set:
            raise ValueError("Test set is empty. Annotate queries first.")
        
        hits = 0
        for item in self.test_set:
            query = item["query"]
            relevant_ids = set(item["relevant_chunk_ids"])
            
            # Retrieve top-k
            retrieved = self.rag.query(query, top_k=k)
            
            # Check if any relevant chunk was retrieved
            # (In practice, you'd need to track chunk IDs through retrieval)
            if any(rid in retrieved for rid in relevant_ids):
                hits += 1
        
        return hits / len(self.test_set)
    
    def precision_at_k(self, k: int = 5) -> float:
        """Precision@k: fraction of retrieved chunks that are relevant."""
        if not self.test_set:
            raise ValueError("Test set is empty.")
        
        total_precision = 0
        for item in self.test_set:
            query = item["query"]
            relevant_ids = set(item["relevant_chunk_ids"])
            retrieved = self.rag.query(query, top_k=k)
            
            if retrieved:
                relevant_retrieved = sum(1 for r in retrieved if r in relevant_ids)
                total_precision += relevant_retrieved / len(retrieved)
        
        return total_precision / len(self.test_set)
    
    def run_full_evaluation(self) -> Dict:
        """Run all metrics and return results."""
        return {
            "recall@5": self.recall_at_k(5),
            "recall@10": self.recall_at_k(10),
            "precision@5": self.precision_at_k(5),
            "num_queries": len(self.test_set)
        }


# ─────────────────────────────────────────
# HOW TO USE (once test set is annotated)
# ─────────────────────────────────────────
"""
evaluator = RetrievalEvaluator(rag_system)
evaluator.load_test_set("ground_truth.json")
results = evaluator.run_full_evaluation()
print(results)
"""
