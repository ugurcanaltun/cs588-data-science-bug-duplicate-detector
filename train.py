"""
Training script for duplicate bug detection with contrastive learning.

This script trains an SBERT-based model using supervised contrastive loss
to learn embeddings where duplicate bug reports are close together.
"""

import os
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from src.model import BugReportEncoder
from src.loss import SupervisedContrastiveLoss
from src.data import create_dataloader
from src.metrics import RetrievalMetrics, compute_similarity_matrix


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train SBERT for duplicate bug detection")

    # Data arguments
    parser.add_argument('--train_data', type=str, required=True,
                        help='Path to training data file (JSON or CSV)')
    parser.add_argument('--val_data', type=str, required=True,
                        help='Path to validation data file (JSON or CSV)')
    parser.add_argument('--use_vlm', action='store_true',
                        help='Use VLM-augmented text (approach 1). If not set, use text-only (approach 2)')

    # Model arguments
    parser.add_argument('--model_name', type=str,
                        default='sentence-transformers/all-MiniLM-L6-v2',
                        help='Pre-trained SBERT model name')
    parser.add_argument('--freeze', action='store_true',
                        help='Freeze encoder (baseline mode, no fine-tuning)')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--samples_per_cluster', type=int, default=4,
                        help='Number of samples per cluster in each batch')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=1,
                        help='Number of warmup epochs')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='Temperature for contrastive loss')

    # Evaluation arguments
    parser.add_argument('--eval_every', type=int, default=1,
                        help='Evaluate every N epochs')
    parser.add_argument('--eval_batch_size', type=int, default=64,
                        help='Batch size for evaluation')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name (auto-generated if not provided)')
    parser.add_argument('--save_best_only', action='store_true',
                        help='Only save the best checkpoint based on validation recall@10')

    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')

    args = parser.parse_args()

    # Auto-generate experiment name if not provided
    if args.experiment_name is None:
        vlm_suffix = "with_vlm" if args.use_vlm else "without_vlm"
        freeze_suffix = "frozen" if args.freeze else "finetuned"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"{freeze_suffix}_{vlm_suffix}_{timestamp}"

    return args


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    model: nn.Module,
    dataloader,
    criterion,
    optimizer,
    device: str,
    epoch: int
) -> dict:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_valid_anchors = 0
    total_avg_positives = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for texts, cluster_ids, _ in pbar:
        # Move cluster IDs to device
        cluster_ids = cluster_ids.to(device)

        # Forward pass
        embeddings = model(texts)

        # Compute loss
        loss, metrics = criterion(embeddings, cluster_ids)

        # Skip batch if no valid anchors
        if metrics['num_valid_anchors'] == 0:
            continue

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        total_loss += metrics['loss']
        total_valid_anchors += metrics['num_valid_anchors']
        total_avg_positives += metrics['avg_num_positives']
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{metrics['loss']:.4f}",
            'anchors': metrics['num_valid_anchors'],
            'avg_pos': f"{metrics['avg_num_positives']:.1f}"
        })

    # Compute epoch metrics
    epoch_metrics = {
        'loss': total_loss / num_batches if num_batches > 0 else 0.0,
        'avg_valid_anchors': total_valid_anchors / num_batches if num_batches > 0 else 0.0,
        'avg_positives': total_avg_positives / num_batches if num_batches > 0 else 0.0,
    }

    return epoch_metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader,
    device: str,
    metrics_calculator: RetrievalMetrics
) -> dict:
    """Evaluate model on validation set."""
    model.eval()

    all_embeddings = []
    all_cluster_ids = []
    all_bug_ids = []

    # Encode all validation reports
    for texts, cluster_ids, bug_ids in tqdm(dataloader, desc="Encoding validation data"):
        embeddings = model(texts)
        all_embeddings.append(embeddings.cpu())
        all_cluster_ids.extend(cluster_ids.tolist())
        all_bug_ids.extend(bug_ids)

    # Concatenate all embeddings
    all_embeddings = torch.cat(all_embeddings, dim=0)

    # Compute similarity matrix
    similarity_matrix = compute_similarity_matrix(all_embeddings, all_embeddings)

    # Compute retrieval metrics
    metrics = metrics_calculator.compute_metrics(
        similarity_matrix,
        all_cluster_ids,
        all_bug_ids
    )

    return metrics


def save_checkpoint(
    model: nn.Module,
    optimizer,
    epoch: int,
    metrics: dict,
    checkpoint_path: str
):
    """Save model checkpoint."""
    # Save model using SBERT's save method
    model.save_pretrained(checkpoint_path)

    # Save additional training state
    training_state = {
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(training_state, os.path.join(checkpoint_path, 'training_state.pt'))

    logger.info(f"Saved checkpoint to {checkpoint_path}")


def main():
    args = parse_args()
    set_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    logger.info(f"Saved configuration to {config_path}")

    # Setup device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Create dataloaders
    logger.info("Loading training data...")
    train_loader = create_dataloader(
        data_path=args.train_data,
        batch_size=args.batch_size,
        use_vlm_augmentation=args.use_vlm,
        use_cluster_sampling=True,
        samples_per_cluster=args.samples_per_cluster,
        num_workers=args.num_workers
    )

    logger.info("Loading validation data...")
    val_loader = create_dataloader(
        data_path=args.val_data,
        batch_size=args.eval_batch_size,
        use_vlm_augmentation=args.use_vlm,
        use_cluster_sampling=False,
        shuffle=False,
        num_workers=args.num_workers
    )

    # Initialize model
    logger.info(f"Initializing model: {args.model_name}")
    model = BugReportEncoder(
        model_name=args.model_name,
        freeze=args.freeze,
        max_length=args.max_length
    ).to(device)

    # Initialize loss
    criterion = SupervisedContrastiveLoss(temperature=args.temperature)

    # Initialize optimizer and scheduler
    if not args.freeze:
        optimizer = AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        # Warmup + cosine annealing scheduler
        total_steps = len(train_loader) * args.epochs
        warmup_steps = len(train_loader) * args.warmup_epochs

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
    else:
        optimizer = None
        scheduler = None
        logger.info("Model is frozen, skipping optimizer initialization")

    # Initialize metrics calculator
    metrics_calculator = RetrievalMetrics(k_values=[1, 5, 10, 20])

    # Training loop
    best_recall_at_10 = 0.0
    training_history = []

    logger.info("Starting training...")
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")

        # Train
        if not args.freeze:
            train_metrics = train_epoch(
                model, train_loader, criterion, optimizer, device, epoch
            )
            logger.info(f"Train loss: {train_metrics['loss']:.4f}")
        else:
            logger.info("Skipping training (model is frozen)")
            train_metrics = {'loss': 0.0}

        # Evaluate
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            logger.info("Evaluating on validation set...")
            val_metrics = evaluate(model, val_loader, device, metrics_calculator)

            logger.info(metrics_calculator.format_metrics(val_metrics))

            # Save training history
            history_entry = {
                'epoch': epoch,
                'train': train_metrics,
                'val': val_metrics
            }
            training_history.append(history_entry)

            # Save checkpoint
            if args.save_best_only:
                # Save only if this is the best model
                if val_metrics.get('recall@10', 0.0) > best_recall_at_10:
                    best_recall_at_10 = val_metrics['recall@10']
                    checkpoint_path = output_dir / 'best_model'
                    save_checkpoint(model, optimizer, epoch, val_metrics, str(checkpoint_path))
                    logger.info(f"New best model! Recall@10: {best_recall_at_10:.4f}")
            else:
                # Save every checkpoint
                checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}'
                save_checkpoint(model, optimizer, epoch, val_metrics, str(checkpoint_path))

        # Step scheduler
        if scheduler is not None:
            scheduler.step()

    # Save final model
    final_checkpoint_path = output_dir / 'final_model'
    save_checkpoint(model, optimizer, args.epochs, val_metrics, str(final_checkpoint_path))

    # Save training history
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    logger.info(f"Saved training history to {history_path}")

    logger.info(f"\nTraining completed! Best Recall@10: {best_recall_at_10:.4f}")
    logger.info(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
