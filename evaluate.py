"""
Evaluation script for duplicate bug detection retrieval task.

This script evaluates a trained SBERT model on the test set using retrieval metrics:
Recall@k, MRR, and MAP@k.
"""

import os
import argparse
import json
import logging
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

from src.model import BugReportEncoder
from src.data import load_data_for_evaluation
from src.metrics import RetrievalMetrics, compute_similarity_matrix


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate SBERT for duplicate bug detection")

    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--baseline', action='store_true',
                        help='Evaluate baseline (frozen pre-trained model)')

    # Data arguments
    parser.add_argument('--test_data', type=str, required=True,
                        help='Path to test data file (JSON or CSV)')
    parser.add_argument('--use_vlm', action='store_true',
                        help='Use VLM-augmented text (approach 1). If not set, use text-only (approach 2)')

    # Evaluation arguments
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for encoding')
    parser.add_argument('--k_values', type=int, nargs='+', default=[1, 5, 10, 20],
                        help='K values for Recall@k and MAP@k')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results (default: same as model_path)')
    parser.add_argument('--save_embeddings', action='store_true',
                        help='Save test embeddings to disk')
    parser.add_argument('--save_similarities', action='store_true',
                        help='Save similarity matrix to disk (warning: can be large)')

    # Other arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for evaluation')

    return parser.parse_args()


@torch.no_grad()
def encode_reports(
    model: BugReportEncoder,
    texts: list,
    batch_size: int,
    device: str
) -> torch.Tensor:
    """
    Encode all bug reports into embeddings.

    Args:
        model: Trained bug report encoder
        texts: List of augmented bug report texts
        batch_size: Batch size for encoding
        device: Device to use

    Returns:
        Tensor of embeddings (num_reports, embedding_dim)
    """
    model.eval()
    all_embeddings = []

    # Process in batches
    num_batches = (len(texts) + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc="Encoding reports"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(texts))
        batch_texts = texts[start_idx:end_idx]

        # Encode batch
        embeddings = model(batch_texts)
        all_embeddings.append(embeddings.cpu())

    # Concatenate all embeddings
    all_embeddings = torch.cat(all_embeddings, dim=0)

    return all_embeddings


def evaluate_retrieval(
    embeddings: torch.Tensor,
    cluster_ids: list,
    bug_ids: list,
    k_values: list
) -> dict:
    """
    Evaluate retrieval performance.

    Args:
        embeddings: Embeddings tensor (num_reports, embedding_dim)
        cluster_ids: List of duplicate cluster IDs
        bug_ids: List of bug report IDs
        k_values: List of k values for metrics

    Returns:
        Dictionary of computed metrics
    """
    logger.info("Computing similarity matrix...")
    similarity_matrix = compute_similarity_matrix(embeddings, embeddings)

    logger.info("Computing retrieval metrics...")
    metrics_calculator = RetrievalMetrics(k_values=k_values)
    metrics = metrics_calculator.compute_metrics(
        similarity_matrix,
        cluster_ids,
        bug_ids
    )

    return metrics, similarity_matrix


def main():
    args = parse_args()

    # Setup device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Load model
    logger.info(f"Loading model from {args.model_path}")
    if args.baseline:
        # For baseline, load pre-trained SBERT directly
        model = BugReportEncoder(
            model_name=args.model_path,
            freeze=True
        ).to(device)
        logger.info("Loaded baseline (frozen) model")
    else:
        # Load fine-tuned model
        model = BugReportEncoder.load_pretrained(
            model_path=args.model_path,
            freeze=False
        ).to(device)
        logger.info("Loaded fine-tuned model")

    # Load test data
    logger.info(f"Loading test data from {args.test_data}")
    texts, cluster_ids, bug_ids = load_data_for_evaluation(
        data_path=args.test_data,
        use_vlm_augmentation=args.use_vlm
    )
    logger.info(f"Loaded {len(texts)} test reports")

    # Count duplicate clusters
    unique_clusters = set(cluster_ids)
    logger.info(f"Number of unique duplicate clusters: {len(unique_clusters)}")

    # Encode all reports
    logger.info("Encoding test reports...")
    embeddings = encode_reports(model, texts, args.batch_size, device)
    logger.info(f"Embeddings shape: {embeddings.shape}")

    # Evaluate retrieval
    metrics, similarity_matrix = evaluate_retrieval(
        embeddings, cluster_ids, bug_ids, args.k_values
    )

    # Print results
    metrics_calculator = RetrievalMetrics(k_values=args.k_values)
    logger.info(metrics_calculator.format_metrics(metrics))

    # Determine output directory
    if args.output_dir is None:
        output_dir = Path(args.model_path).parent / 'evaluation_results'
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    vlm_suffix = "with_vlm" if args.use_vlm else "without_vlm"
    metrics_file = output_dir / f'metrics_{vlm_suffix}.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_file}")

    # Save embeddings if requested
    if args.save_embeddings:
        embeddings_file = output_dir / f'embeddings_{vlm_suffix}.pt'
        torch.save({
            'embeddings': embeddings,
            'cluster_ids': cluster_ids,
            'bug_ids': bug_ids
        }, embeddings_file)
        logger.info(f"Saved embeddings to {embeddings_file}")

    # Save similarity matrix if requested
    if args.save_similarities:
        similarities_file = output_dir / f'similarities_{vlm_suffix}.pt'
        torch.save(similarity_matrix, similarities_file)
        logger.info(f"Saved similarity matrix to {similarities_file}")

    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Test data: {args.test_data}")
    logger.info(f"VLM augmentation: {args.use_vlm}")
    logger.info(f"Number of test reports: {len(texts)}")
    logger.info(f"Number of queries with duplicates: {int(metrics.get('num_queries', 0))}")
    logger.info("")
    logger.info(f"Recall@1:  {metrics.get('recall@1', 0):.4f}")
    logger.info(f"Recall@5:  {metrics.get('recall@5', 0):.4f}")
    logger.info(f"Recall@10: {metrics.get('recall@10', 0):.4f}")
    logger.info(f"Recall@20: {metrics.get('recall@20', 0):.4f}")
    logger.info("")
    logger.info(f"MRR: {metrics.get('mrr', 0):.4f}")
    logger.info("")
    logger.info(f"MAP@1:  {metrics.get('map@1', 0):.4f}")
    logger.info(f"MAP@5:  {metrics.get('map@5', 0):.4f}")
    logger.info(f"MAP@10: {metrics.get('map@10', 0):.4f}")
    logger.info(f"MAP@20: {metrics.get('map@20', 0):.4f}")
    logger.info("=" * 50)


if __name__ == '__main__':
    main()
