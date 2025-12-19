"""
Evaluation metrics for duplicate bug detection retrieval task.

This module implements Recall@k, Mean Reciprocal Rank (MRR), and Mean Average
Precision at k (MAP@k) for evaluating the quality of duplicate bug retrieval.
"""

import torch
import numpy as np
from typing import List, Dict, Set
from collections import defaultdict


class RetrievalMetrics:
    """
    Compute retrieval metrics for duplicate bug detection.

    Metrics computed:
    - Recall@k: Proportion of queries where at least one duplicate appears in top-k
    - MRR: Mean reciprocal rank of the first relevant item
    - MAP@k: Mean average precision at k, rewarding systems that rank all duplicates highly
    """

    def __init__(self, k_values: List[int] = [1, 5, 10, 20]):
        """
        Initialize metrics calculator.

        Args:
            k_values: List of k values for which to compute Recall@k and MAP@k
        """
        self.k_values = sorted(k_values)

    def compute_recall_at_k(
        self,
        ranked_candidates: List[int],
        relevant_ids: Set[int],
        k: int
    ) -> float:
        """
        Compute Recall@k for a single query.

        Recall@k = 1 if at least one relevant item appears in top-k, else 0

        Args:
            ranked_candidates: List of candidate IDs ranked by similarity (descending)
            relevant_ids: Set of relevant (duplicate) candidate IDs
            k: Cutoff position

        Returns:
            1.0 if at least one relevant item in top-k, else 0.0
        """
        top_k = set(ranked_candidates[:k])
        return 1.0 if len(top_k & relevant_ids) > 0 else 0.0

    def compute_reciprocal_rank(
        self,
        ranked_candidates: List[int],
        relevant_ids: Set[int]
    ) -> float:
        """
        Compute reciprocal rank for a single query.

        RR = 1 / rank_of_first_relevant_item

        Args:
            ranked_candidates: List of candidate IDs ranked by similarity (descending)
            relevant_ids: Set of relevant (duplicate) candidate IDs

        Returns:
            Reciprocal of the rank of the first relevant item (0.0 if no relevant items)
        """
        for rank, candidate_id in enumerate(ranked_candidates, start=1):
            if candidate_id in relevant_ids:
                return 1.0 / rank
        return 0.0

    def compute_average_precision_at_k(
        self,
        ranked_candidates: List[int],
        relevant_ids: Set[int],
        k: int
    ) -> float:
        """
        Compute Average Precision at k for a single query.

        AP@k = (1/|D_q|) * sum_{i=1}^{k} (1[c_i in D_q] * P@i)

        where P@i is the precision at position i.

        Args:
            ranked_candidates: List of candidate IDs ranked by similarity (descending)
            relevant_ids: Set of relevant (duplicate) candidate IDs
            k: Cutoff position

        Returns:
            Average precision at k
        """
        if len(relevant_ids) == 0:
            return 0.0

        num_relevant = len(relevant_ids)
        top_k = ranked_candidates[:k]

        precision_sum = 0.0
        num_hits = 0

        for i, candidate_id in enumerate(top_k, start=1):
            if candidate_id in relevant_ids:
                num_hits += 1
                precision_at_i = num_hits / i
                precision_sum += precision_at_i

        return precision_sum / num_relevant

    def compute_metrics_single_query(
        self,
        ranked_candidates: List[int],
        relevant_ids: Set[int]
    ) -> Dict[str, float]:
        """
        Compute all metrics for a single query.

        Args:
            ranked_candidates: List of candidate IDs ranked by similarity (descending)
            relevant_ids: Set of relevant (duplicate) candidate IDs

        Returns:
            Dictionary containing all computed metrics
        """
        metrics = {}

        # Compute Recall@k for all k values
        for k in self.k_values:
            metrics[f"recall@{k}"] = self.compute_recall_at_k(
                ranked_candidates, relevant_ids, k
            )

        # Compute MRR
        metrics["mrr"] = self.compute_reciprocal_rank(ranked_candidates, relevant_ids)

        # Compute MAP@k for all k values
        for k in self.k_values:
            metrics[f"map@{k}"] = self.compute_average_precision_at_k(
                ranked_candidates, relevant_ids, k
            )

        return metrics

    def compute_metrics(
        self,
        similarities: torch.Tensor,
        duplicate_clusters: List[int],
        candidate_ids: List[int]
    ) -> Dict[str, float]:
        """
        Compute metrics for all queries in a test set.

        Args:
            similarities: Similarity matrix of shape (num_queries, num_candidates)
                         where similarities[i, j] is the similarity between query i and candidate j
            duplicate_clusters: List of duplicate cluster IDs for each query/candidate
            candidate_ids: List of candidate bug report IDs

        Returns:
            Dictionary containing mean metrics across all queries
        """
        num_queries = similarities.shape[0]

        # Convert to numpy for easier processing
        if isinstance(similarities, torch.Tensor):
            similarities = similarities.cpu().numpy()

        # Aggregate metrics across all queries
        all_metrics = defaultdict(list)

        for query_idx in range(num_queries):
            query_cluster = duplicate_clusters[query_idx]
            query_id = candidate_ids[query_idx]

            # Find all relevant candidates (same cluster, excluding query itself)
            relevant_ids = set()
            for cand_idx, cand_cluster in enumerate(duplicate_clusters):
                cand_id = candidate_ids[cand_idx]
                if cand_cluster == query_cluster and cand_id != query_id:
                    relevant_ids.add(cand_id)

            # Skip queries with no duplicates
            if len(relevant_ids) == 0:
                continue

            # Get similarity scores for this query
            query_similarities = similarities[query_idx]

            # Rank candidates by similarity (descending order)
            # Exclude the query itself from candidates
            ranked_indices = np.argsort(-query_similarities)
            ranked_candidates = [
                candidate_ids[idx] for idx in ranked_indices
                if candidate_ids[idx] != query_id
            ]

            # Compute metrics for this query
            query_metrics = self.compute_metrics_single_query(
                ranked_candidates, relevant_ids
            )

            # Aggregate
            for metric_name, value in query_metrics.items():
                all_metrics[metric_name].append(value)

        # Compute mean metrics
        mean_metrics = {}
        for metric_name, values in all_metrics.items():
            mean_metrics[metric_name] = np.mean(values) if len(values) > 0 else 0.0

        # Add number of queries
        mean_metrics["num_queries"] = len(all_metrics.get("mrr", []))

        return mean_metrics

    def format_metrics(self, metrics: Dict[str, float]) -> str:
        """
        Format metrics for pretty printing.

        Args:
            metrics: Dictionary of computed metrics

        Returns:
            Formatted string representation
        """
        lines = ["\nRetrieval Metrics:"]
        lines.append("=" * 50)

        # Print Recall@k
        lines.append("\nRecall@k:")
        for k in self.k_values:
            if f"recall@{k}" in metrics:
                lines.append(f"  Recall@{k:2d}: {metrics[f'recall@{k}']:.4f}")

        # Print MRR
        if "mrr" in metrics:
            lines.append(f"\nMRR: {metrics['mrr']:.4f}")

        # Print MAP@k
        lines.append("\nMAP@k:")
        for k in self.k_values:
            if f"map@{k}" in metrics:
                lines.append(f"  MAP@{k:2d}: {metrics[f'map@{k}']:.4f}")

        # Print number of queries
        if "num_queries" in metrics:
            lines.append(f"\nNumber of queries: {int(metrics['num_queries'])}")

        lines.append("=" * 50)

        return "\n".join(lines)


def compute_similarity_matrix(
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor
) -> torch.Tensor:
    """
    Compute cosine similarity matrix between query and candidate embeddings.

    Args:
        query_embeddings: Embeddings of shape (num_queries, embedding_dim)
        candidate_embeddings: Embeddings of shape (num_candidates, embedding_dim)

    Returns:
        Similarity matrix of shape (num_queries, num_candidates)
    """
    # Ensure embeddings are L2-normalized
    query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
    candidate_embeddings = torch.nn.functional.normalize(candidate_embeddings, p=2, dim=1)

    # Compute cosine similarity via dot product
    similarity_matrix = torch.matmul(query_embeddings, candidate_embeddings.T)

    return similarity_matrix
