"""
Training script for duplicate bug detection with contrastive learning.

This script trains an SBERT-based model using supervised contrastive loss
to learn embeddings where duplicate bug reports are close together.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
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


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train SBERT for duplicate bug detection")

    # Config file argument
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config JSON file. CLI arguments override config file values.')

    args = parser.parse_args()

    # Load config file if provided
    if args.config is not None:
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
        
        args.experiment_name = config['experiment_name']
        args.model_name = config['model']['model_name']
        
        args.train_data = config['data']['train_data']
        args.val_data = config['data']['val_data']
        args.use_vlm = config['data']['use_vlm_augmentation']
        
        args.batch_size = config['training']['batch_size']
        args.samples_per_cluster = config['training']['samples_per_cluster']
        args.epochs = config['training']['epochs']
        args.lr = config['training']['learning_rate']
        args.weight_decay = config['training']['weight_decay']
        args.warmup_epochs = config['training']['warmup_epochs']
        args.temperature = config['training']['temperature']
        
        args.eval_every = config['evaluation']['eval_every']
        args.eval_batch_size = config['evaluation']['eval_batch_size']
        args.k_values = config['evaluation']['k_values']
        
        args.output_dir = config['output']['output_dir']
        args.save_best_only = config['output']['save_best_only']
        
        args.device = config['hardware']['device']
        args.num_workers = config['hardware']['num_workers']

        args.seed = config['seed']

        # Add suffix to experiment name based on VLM usage
        vlm_suffix = "_with_vlm" if args.use_vlm else "_without_vlm"
        args.experiment_name = f"{args.experiment_name}{vlm_suffix}"
    else:
        parser.error("Config file is required. Please provide --config argument.")

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
    epoch: int,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
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
        
        if scheduler is not None:
            scheduler.step()

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
        freeze=False,
    ).to(device)

    # Initialize loss
    criterion = SupervisedContrastiveLoss(temperature=args.temperature)

    # Initialize optimizer and scheduler
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


    # Initialize metrics calculator
    metrics_calculator = RetrievalMetrics(args.k_values)

    # Training loop
    best_recall_at_10 = 0.0
    training_history = []

    logger.info("Starting training...")
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            scheduler=scheduler
        )
        logger.info(f"Train loss: {train_metrics['loss']:.4f}")

        # Evaluate on validation set after every epoch
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

        # Save checkpoint (controlled by eval_every and save_best_only)
        should_save_checkpoint = (epoch % args.eval_every == 0 or epoch == args.epochs)

        if should_save_checkpoint:
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
