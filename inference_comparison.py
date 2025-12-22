#!/usr/bin/env python3
"""
Inference Comparison Script for Bug Duplicate Detection

This script:
1. Loads a fine-tuned model
2. Takes a sample bug report with attachments (visual context filled)
3. Performs inference with VLM outputs and without VLM outputs
4. Retrieves top 20 similar bug reports for each approach
5. Indicates which retrieved reports are actual duplicates
"""

import argparse
import json
import torch
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

from src.model import BugReportEncoder
from src.metrics import compute_similarity_matrix


def load_test_data(data_path: str) -> Tuple[pd.DataFrame, List[str], List[str], List[int], List[str]]:
    """
    Load test data with both VLM and non-VLM versions.

    Returns:
        df: DataFrame with all data
        texts_with_vlm: Bug reports with VLM augmentation
        texts_without_vlm: Bug reports without VLM augmentation
        cluster_ids: Duplicate cluster IDs
        bug_ids: Bug report IDs
    """
    df = pd.read_csv(data_path)

    texts_with_vlm = df["augmented_text_with_vlm"].tolist()
    texts_without_vlm = df["augmented_text_without_vlm"].tolist()
    cluster_ids = df["duplicate_cluster_id"].tolist()
    bug_ids = df["bug_id"].tolist()

    return df, texts_with_vlm, texts_without_vlm, cluster_ids, bug_ids


def get_duplicates_for_query(query_idx: int, cluster_ids: List[int], bug_ids: List[str]) -> List[str]:
    """
    Get all bug IDs that are duplicates of the query bug.

    Args:
        query_idx: Index of the query bug
        cluster_ids: List of duplicate cluster IDs
        bug_ids: List of bug IDs

    Returns:
        List of bug IDs that are duplicates of the query
    """
    query_cluster = cluster_ids[query_idx]
    query_bug_id = bug_ids[query_idx]

    duplicates = []
    for idx, (cluster_id, bug_id) in enumerate(zip(cluster_ids, bug_ids)):
        if cluster_id == query_cluster and bug_id != query_bug_id:
            duplicates.append(bug_id)

    return duplicates


def retrieve_top_k(query_idx: int, similarity_scores: np.ndarray, bug_ids: List[str], k: int = 20) -> List[Tuple[str, float, int]]:
    """
    Retrieve top-k similar bug reports for a query.

    Args:
        query_idx: Index of the query bug
        similarity_scores: Similarity scores matrix row for the query
        bug_ids: List of bug IDs
        k: Number of results to retrieve

    Returns:
        List of (bug_id, similarity_score, rank) tuples
    """
    # Get indices sorted by similarity (descending)
    sorted_indices = np.argsort(similarity_scores)[::-1]

    # Remove the query itself from results
    query_bug_id = bug_ids[query_idx]
    sorted_indices = [idx for idx in sorted_indices if bug_ids[idx] != query_bug_id]

    # Get top-k
    top_k_indices = sorted_indices[:k]

    results = []
    for rank, idx in enumerate(top_k_indices, start=1):
        results.append((bug_ids[idx], similarity_scores[idx], rank))

    return results


def print_comparison_results(
    query_idx: int,
    query_bug_id: str,
    query_text_with_vlm: str,
    query_text_without_vlm: str,
    duplicates: List[str],
    results_with_vlm: List[Tuple[str, float, int]],
    results_without_vlm: List[Tuple[str, float, int]]
):
    """Print formatted comparison results."""

    print("=" * 100)
    print(f"QUERY BUG REPORT: {query_bug_id}")
    print("=" * 100)
    print("\n--- Query Text (WITHOUT VLM) ---")
    print(query_text_without_vlm[:500] + "..." if len(query_text_without_vlm) > 500 else query_text_without_vlm)

    print("\n--- Query Text (WITH VLM) ---")
    print(query_text_with_vlm[:500] + "..." if len(query_text_with_vlm) > 500 else query_text_with_vlm)

    print(f"\n--- Known Duplicates: {len(duplicates)} ---")
    print(", ".join(duplicates[:10]) + ("..." if len(duplicates) > 10 else ""))

    # Results WITHOUT VLM
    print("\n" + "=" * 100)
    print("TOP 20 RESULTS - WITHOUT VLM")
    print("=" * 100)
    print(f"{'Rank':<6} {'Bug ID':<15} {'Similarity':<12} {'Duplicate?':<12}")
    print("-" * 100)

    duplicates_found_without = 0
    for bug_id, similarity, rank in results_without_vlm:
        is_duplicate = "✓ DUPLICATE" if bug_id in duplicates else ""
        if is_duplicate:
            duplicates_found_without += 1
        print(f"{rank:<6} {bug_id:<15} {similarity:<12.4f} {is_duplicate:<12}")

    # Results WITH VLM
    print("\n" + "=" * 100)
    print("TOP 20 RESULTS - WITH VLM")
    print("=" * 100)
    print(f"{'Rank':<6} {'Bug ID':<15} {'Similarity':<12} {'Duplicate?':<12}")
    print("-" * 100)

    duplicates_found_with = 0
    for bug_id, similarity, rank in results_with_vlm:
        is_duplicate = "✓ DUPLICATE" if bug_id in duplicates else ""
        if is_duplicate:
            duplicates_found_with += 1
        print(f"{rank:<6} {bug_id:<15} {similarity:<12.4f} {is_duplicate:<12}")

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"Total known duplicates: {len(duplicates)}")
    print(f"Duplicates found in top-20 WITHOUT VLM: {duplicates_found_without}/{len(duplicates)} ({100*duplicates_found_without/len(duplicates):.1f}%)")
    print(f"Duplicates found in top-20 WITH VLM: {duplicates_found_with}/{len(duplicates)} ({100*duplicates_found_with/len(duplicates):.1f}%)")
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Compare inference with and without VLM outputs")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned model WITH VLM (e.g., outputs/sbert_contrastive_with_vlm/best_model)"
    )
    parser.add_argument(
        "--model_path_without_vlm",
        type=str,
        required=True,
        help="Path to the fine-tuned model WITHOUT VLM (e.g., outputs/sbert_contrastive_without_vlm/best_model)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/processed/test.csv",
        help="Path to test data CSV file"
    )
    parser.add_argument(
        "--query_idx",
        type=int,
        default=None,
        help="Index of the query bug report (if not specified, will select a sample with attachments)"
    )
    parser.add_argument(
        "--query_bug_id",
        type=str,
        default=None,
        help="Bug ID to use as query (alternative to --query_idx)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for encoding"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for inference"
    )

    args = parser.parse_args()

    print(f"Loading model WITH VLM from: {args.model_path}")
    model_with_vlm = BugReportEncoder.load_pretrained(
        model_path=args.model_path,
        freeze=True  # Freeze for inference
    ).to(args.device)
    model_with_vlm.eval()

    print(f"Loading model WITHOUT VLM from: {args.model_path_without_vlm}")
    model_without_vlm = BugReportEncoder.load_pretrained(
        model_path=args.model_path_without_vlm,
        freeze=True  # Freeze for inference
    ).to(args.device)
    model_without_vlm.eval()

    print(f"Loading test data from: {args.data_path}")
    df, texts_with_vlm, texts_without_vlm, cluster_ids, bug_ids = load_test_data(args.data_path)

    # Select query sample
    if args.query_bug_id:
        # Find by bug ID
        try:
            query_idx = bug_ids.index(args.query_bug_id)
        except ValueError:
            print(f"Error: Bug ID {args.query_bug_id} not found in dataset")
            return
    elif args.query_idx is not None:
        query_idx = args.query_idx
    else:
        # Find a sample with visual context (has VLM outputs)
        print("Searching for a sample with visual context (VLM outputs)...")
        for idx in range(len(texts_with_vlm)):
            # Check if there's actual content after "Visual Context:"
            if "Visual Context:" in texts_with_vlm[idx]:
                visual_context_start = texts_with_vlm[idx].find("Visual Context:")
                content_after = texts_with_vlm[idx][visual_context_start + len("Visual Context:"):].strip()
                # Check if there's meaningful content (not empty or just whitespace)
                if content_after and content_after != "":
                    # Also check that it has duplicates
                    duplicates = get_duplicates_for_query(idx, cluster_ids, bug_ids)
                    if len(duplicates) > 0:
                        query_idx = idx
                        print(f"Found sample at index {idx} with {len(duplicates)} duplicates")
                        break
        else:
            print("Warning: No sample with visual context found. Using first sample with duplicates.")
            for idx in range(len(cluster_ids)):
                duplicates = get_duplicates_for_query(idx, cluster_ids, bug_ids)
                if len(duplicates) > 0:
                    query_idx = idx
                    break

    query_bug_id = bug_ids[query_idx]
    query_text_with_vlm = texts_with_vlm[query_idx]
    query_text_without_vlm = texts_without_vlm[query_idx]
    duplicates = get_duplicates_for_query(query_idx, cluster_ids, bug_ids)

    if len(duplicates) == 0:
        print(f"Warning: Query bug {query_bug_id} has no duplicates in the dataset")

    # Encode all bug reports WITHOUT VLM using model trained without VLM
    print("\nEncoding all bug reports WITHOUT VLM (using model trained without VLM)...")
    embeddings_without_vlm = model_without_vlm.encode_batch(
        texts_without_vlm,
        batch_size=args.batch_size,
        show_progress=True
    )

    # Encode all bug reports WITH VLM using model trained with VLM
    print("\nEncoding all bug reports WITH VLM (using model trained with VLM)...")
    embeddings_with_vlm = model_with_vlm.encode_batch(
        texts_with_vlm,
        batch_size=args.batch_size,
        show_progress=True
    )

    # Compute similarity matrices
    print("\nComputing similarity matrices...")
    similarity_without_vlm = compute_similarity_matrix(
        embeddings_without_vlm[query_idx:query_idx+1],
        embeddings_without_vlm
    )[0]  # Get first row (only one query)

    similarity_with_vlm = compute_similarity_matrix(
        embeddings_with_vlm[query_idx:query_idx+1],
        embeddings_with_vlm
    )[0]  # Get first row (only one query)

    # Retrieve top-20 results
    results_without_vlm = retrieve_top_k(query_idx, similarity_without_vlm, bug_ids, k=20)
    results_with_vlm = retrieve_top_k(query_idx, similarity_with_vlm, bug_ids, k=20)

    # Print comparison
    print_comparison_results(
        query_idx,
        query_bug_id,
        query_text_with_vlm,
        query_text_without_vlm,
        duplicates,
        results_with_vlm,
        results_without_vlm
    )


if __name__ == "__main__":
    main()
