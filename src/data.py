"""
Data loading utilities for duplicate bug detection.

This module provides dataset classes and utilities for loading augmented bug reports
and creating batches for contrastive learning.
"""

import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import random
import numpy as np


class BugReportDataset(Dataset):
    """
    Dataset for bug reports with augmented text.

    Expected data format (JSON or CSV):
    - bug_id: Unique identifier for the bug report
    - augmented_text: The augmented text combining all fields and VLM outputs
    - duplicate_cluster_id: Cluster ID for duplicate grouping

    Args:
        data_path: Path to the data file (JSON or CSV)
        use_vlm_augmentation: If False, load text without VLM-generated sections
    """

    def __init__(
        self,
        data_path: str,
        use_vlm_augmentation: bool = True
    ):
        self.data_path = data_path
        self.use_vlm_augmentation = use_vlm_augmentation

        # Load data
        self.data = self._load_data()

        # Create mappings
        self._create_cluster_mapping()

    def _load_data(self) -> pd.DataFrame:
        """Load data from file."""
        if self.data_path.endswith('.json'):
            with open(self.data_path, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        elif self.data_path.endswith('.csv'):
            df = pd.read_csv(self.data_path)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path}")

        # Validate required columns
        required_cols = ['bug_id', 'duplicate_cluster_id']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Select appropriate text column
        if self.use_vlm_augmentation:
            if 'augmented_text_with_vlm' in df.columns:
                df['text'] = df['augmented_text_with_vlm']
            elif 'augmented_text' in df.columns:
                df['text'] = df['augmented_text']
            else:
                raise ValueError("No augmented text column found")
        else:
            if 'augmented_text_without_vlm' in df.columns:
                df['text'] = df['augmented_text_without_vlm']
            elif 'augmented_text' in df.columns:
                df['text'] = df['augmented_text']
            else:
                raise ValueError("No augmented text column found")

        return df

    def _create_cluster_mapping(self):
        """Create mapping from cluster IDs to bug report indices."""
        self.cluster_to_indices = defaultdict(list)
        for idx, cluster_id in enumerate(self.data['duplicate_cluster_id']):
            self.cluster_to_indices[cluster_id].append(idx)

        # Get list of clusters with multiple reports (for training)
        self.duplicate_clusters = [
            cluster_id for cluster_id, indices in self.cluster_to_indices.items()
            if len(indices) > 1
        ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single bug report.

        Returns:
            Dictionary with keys: bug_id, text, cluster_id
        """
        row = self.data.iloc[idx]
        return {
            'bug_id': row['bug_id'],
            'text': row['text'],
            'cluster_id': row['duplicate_cluster_id'],
            'index': idx
        }

    def get_bug_ids(self) -> List:
        """Get list of all bug IDs."""
        return self.data['bug_id'].tolist()

    def get_cluster_ids(self) -> List:
        """Get list of all cluster IDs."""
        return self.data['duplicate_cluster_id'].tolist()

    def get_texts(self) -> List[str]:
        """Get list of all augmented texts."""
        return self.data['text'].tolist()


class ClusterBalancedBatchSampler(Sampler):
    """
    Batch sampler that ensures each batch contains multiple samples from the same clusters.

    This is crucial for contrastive learning, as we need positive pairs in each batch.
    The sampler creates batches by:
    1. Sampling a set of clusters
    2. Sampling multiple reports from each selected cluster
    3. Ensuring each batch has sufficient positive pairs

    Args:
        dataset: BugReportDataset instance
        batch_size: Total number of samples per batch
        samples_per_cluster: Number of samples to draw from each cluster
        drop_last: Whether to drop the last incomplete batch
    """

    def __init__(
        self,
        dataset: BugReportDataset,
        batch_size: int = 32,
        samples_per_cluster: int = 4,
        drop_last: bool = True
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.samples_per_cluster = samples_per_cluster
        self.drop_last = drop_last

        # Calculate how many clusters per batch
        self.clusters_per_batch = max(1, batch_size // samples_per_cluster)

        # Get list of clusters that have at least samples_per_cluster reports
        self.valid_clusters = [
            cluster_id for cluster_id, indices in dataset.cluster_to_indices.items()
            if len(indices) >= samples_per_cluster
        ]

        if len(self.valid_clusters) == 0:
            raise ValueError(
                f"No clusters have at least {samples_per_cluster} samples. "
                f"Consider reducing samples_per_cluster."
            )

    def __iter__(self):
        """Generate batches."""
        # Shuffle clusters
        random.shuffle(self.valid_clusters)

        for i in range(0, len(self.valid_clusters), self.clusters_per_batch):
            # Select clusters for this batch
            batch_clusters = self.valid_clusters[i:i + self.clusters_per_batch]

            if len(batch_clusters) < self.clusters_per_batch and self.drop_last:
                continue

            # Sample indices from selected clusters
            batch_indices = []
            for cluster_id in batch_clusters:
                cluster_indices = self.dataset.cluster_to_indices[cluster_id]
                # Sample with replacement if cluster is too small
                sampled = random.choices(
                    cluster_indices,
                    k=min(self.samples_per_cluster, len(cluster_indices))
                )
                batch_indices.extend(sampled)

            # Shuffle within batch
            random.shuffle(batch_indices)

            yield batch_indices

    def __len__(self):
        """Number of batches."""
        n_batches = len(self.valid_clusters) // self.clusters_per_batch
        if not self.drop_last and len(self.valid_clusters) % self.clusters_per_batch != 0:
            n_batches += 1
        return n_batches


def collate_fn(batch: List[Dict]) -> Tuple[List[str], torch.Tensor, List[int]]:
    """
    Collate function for DataLoader.

    Args:
        batch: List of dictionaries from dataset

    Returns:
        Tuple of (texts, cluster_ids_tensor, bug_ids)
    """
    texts = [item['text'] for item in batch]
    cluster_ids = torch.tensor([item['cluster_id'] for item in batch], dtype=torch.long)
    bug_ids = [item['bug_id'] for item in batch]

    return texts, cluster_ids, bug_ids


def create_dataloader(
    data_path: str,
    batch_size: int = 32,
    use_vlm_augmentation: bool = True,
    use_cluster_sampling: bool = True,
    samples_per_cluster: int = 4,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    Create a DataLoader for bug reports.

    Args:
        data_path: Path to data file
        batch_size: Batch size
        use_vlm_augmentation: Whether to use VLM-augmented text
        use_cluster_sampling: Whether to use cluster-balanced sampling for training
        samples_per_cluster: Number of samples per cluster (if using cluster sampling)
        shuffle: Whether to shuffle data (ignored if use_cluster_sampling=True)
        num_workers: Number of data loading workers

    Returns:
        DataLoader instance
    """
    dataset = BugReportDataset(
        data_path=data_path,
        use_vlm_augmentation=use_vlm_augmentation
    )

    if use_cluster_sampling:
        # Use cluster-balanced sampling for contrastive learning
        batch_sampler = ClusterBalancedBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            samples_per_cluster=samples_per_cluster,
            drop_last=True
        )

        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=num_workers
        )
    else:
        # Standard sampling for evaluation
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=num_workers,
            drop_last=False
        )

    return dataloader


def load_data_for_evaluation(
    data_path: str,
    use_vlm_augmentation: bool = True
) -> Tuple[List[str], List[int], List[int]]:
    """
    Load all data for evaluation.

    Args:
        data_path: Path to data file
        use_vlm_augmentation: Whether to use VLM-augmented text

    Returns:
        Tuple of (texts, cluster_ids, bug_ids)
    """
    dataset = BugReportDataset(
        data_path=data_path,
        use_vlm_augmentation=use_vlm_augmentation
    )

    texts = dataset.get_texts()
    cluster_ids = dataset.get_cluster_ids()
    bug_ids = dataset.get_bug_ids()

    return texts, cluster_ids, bug_ids
