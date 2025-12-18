"""
Quick script to evaluate baseline (frozen pre-trained SBERT) without training.

This script directly evaluates a pre-trained SBERT model on test data
without any fine-tuning, serving as a baseline for comparison.
"""

import argparse
import logging

from evaluate import parse_args as eval_parse_args, main as eval_main


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate baseline (frozen pre-trained SBERT) for duplicate bug detection"
    )

    # Model arguments
    parser.add_argument('--model_name', type=str,
                        default='sentence-transformers/all-MiniLM-L6-v2',
                        help='Pre-trained SBERT model name')

    # Data arguments
    parser.add_argument('--test_data', type=str, required=True,
                        help='Path to test data file (JSON or CSV)')
    parser.add_argument('--use_vlm', action='store_true',
                        help='Use VLM-augmented text')

    # Evaluation arguments
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for encoding')
    parser.add_argument('--k_values', type=int, nargs='+', default=[1, 5, 10, 20],
                        help='K values for Recall@k and MAP@k')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='baseline_results',
                        help='Output directory for results')
    parser.add_argument('--save_embeddings', action='store_true',
                        help='Save test embeddings to disk')

    # Other arguments
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for evaluation')

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("BASELINE EVALUATION (Frozen Pre-trained SBERT)")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Test data: {args.test_data}")
    logger.info(f"VLM augmentation: {args.use_vlm}")
    logger.info("=" * 60)

    # Create eval arguments compatible with evaluate.py
    import sys
    eval_args = [
        '--model_path', args.model_name,
        '--baseline',
        '--test_data', args.test_data,
        '--batch_size', str(args.batch_size),
        '--output_dir', args.output_dir,
        '--device', args.device,
    ]

    if args.use_vlm:
        eval_args.append('--use_vlm')

    if args.save_embeddings:
        eval_args.append('--save_embeddings')

    eval_args.extend(['--k_values'] + [str(k) for k in args.k_values])

    # Temporarily replace sys.argv
    original_argv = sys.argv
    sys.argv = ['evaluate.py'] + eval_args

    try:
        # Run evaluation
        eval_main()
    finally:
        # Restore original argv
        sys.argv = original_argv

    logger.info("\n" + "=" * 60)
    logger.info("BASELINE EVALUATION COMPLETED")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
