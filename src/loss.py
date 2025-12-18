"""
Supervised contrastive loss for duplicate bug detection.

This module implements the contrastive learning objective that encourages
embeddings of duplicate bug reports to be close together while pushing
non-duplicates apart.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised contrastive loss for training the bug report encoder.

    For each anchor report i with embedding z_i, let P(i) be the set of indices
    of other reports in the batch that belong to the same duplicate group.
    The loss encourages high similarity between the anchor and its positives
    while maintaining low similarity with negatives.

    Loss formula for anchor i:
        L_i = -1/|P(i)| * sum_{p in P(i)} log(exp(sim(z_i, z_p)) / sum_{a in A(i)} exp(sim(z_i, z_a)))

    where sim(z_i, z_j) = (z_i^T z_j) / tau is the temperature-scaled cosine similarity.

    Args:
        temperature: Temperature parameter for scaling similarities (default: 0.07)
        base_temperature: Base temperature for normalization (default: 0.07)
    """

    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute supervised contrastive loss.

        Args:
            embeddings: Normalized embeddings of shape (batch_size, embedding_dim)
            labels: Duplicate cluster IDs of shape (batch_size,)
                    Reports with the same label belong to the same duplicate group

        Returns:
            loss: Scalar loss value
            metrics: Dictionary containing loss statistics
        """
        device = embeddings.device
        batch_size = embeddings.shape[0]

        # Ensure labels are on the same device
        labels = labels.to(device)

        # Create mask for positive pairs (same duplicate group)
        # Shape: (batch_size, batch_size)
        labels_expanded = labels.unsqueeze(1)  # (batch_size, 1)
        mask = torch.eq(labels_expanded, labels_expanded.T).float()  # (batch_size, batch_size)

        # Compute temperature-scaled cosine similarity matrix
        # Since embeddings are L2-normalized, dot product gives cosine similarity
        similarity_matrix = torch.matmul(embeddings, embeddings.T)  # (batch_size, batch_size)
        similarity_matrix = similarity_matrix / self.temperature

        # For numerical stability, subtract the maximum value
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        # Create mask to exclude self-similarity (diagonal)
        logits_mask = torch.ones_like(mask).scatter_(
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )

        # Mask for positives (excluding self)
        mask = mask * logits_mask

        # Compute log probabilities
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-10)

        # Count positives for each anchor
        num_positives = mask.sum(dim=1)

        # Only compute loss for anchors that have at least one positive
        valid_anchors = num_positives > 0

        if valid_anchors.sum() == 0:
            # No valid anchors in this batch (all reports are from different groups)
            return torch.tensor(0.0, device=device), {
                "loss": 0.0,
                "num_valid_anchors": 0,
                "avg_num_positives": 0.0,
            }

        # Compute mean of log-likelihood over positives for each anchor
        # Shape: (batch_size,)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (num_positives + 1e-10)

        # Apply loss only to valid anchors and compute mean
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss[valid_anchors].mean()

        # Compute statistics for logging
        metrics = {
            "loss": loss.item(),
            "num_valid_anchors": valid_anchors.sum().item(),
            "avg_num_positives": num_positives[valid_anchors].float().mean().item(),
            "avg_similarity_pos": (similarity_matrix * mask).sum() / (mask.sum() + 1e-10),
            "avg_similarity_all": similarity_matrix[logits_mask.bool()].mean().item(),
        }

        return loss, metrics


class TripletLoss(nn.Module):
    """
    Alternative triplet loss for bug report similarity learning.

    This loss can be used as an alternative to supervised contrastive loss.
    For each anchor, it samples one positive and one negative, and enforces:
        distance(anchor, positive) + margin < distance(anchor, negative)

    Args:
        margin: Margin for triplet loss (default: 0.5)
        distance_metric: Distance metric to use ('cosine' or 'euclidean')
    """

    def __init__(self, margin: float = 0.5, distance_metric: str = "cosine"):
        super().__init__()
        self.margin = margin
        self.distance_metric = distance_metric

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute triplet loss.

        Args:
            embeddings: Normalized embeddings of shape (batch_size, embedding_dim)
            labels: Duplicate cluster IDs of shape (batch_size,)

        Returns:
            loss: Scalar loss value
            metrics: Dictionary containing loss statistics
        """
        device = embeddings.device
        batch_size = embeddings.shape[0]

        # Compute pairwise distances
        if self.distance_metric == "cosine":
            # For L2-normalized embeddings, cosine distance = 1 - cosine_similarity
            similarity_matrix = torch.matmul(embeddings, embeddings.T)
            distance_matrix = 1.0 - similarity_matrix
        else:  # euclidean
            # Compute pairwise euclidean distances
            distance_matrix = torch.cdist(embeddings, embeddings, p=2)

        # Create masks for positives and negatives
        labels_expanded = labels.unsqueeze(1)
        positive_mask = torch.eq(labels_expanded, labels_expanded.T).float()
        negative_mask = 1.0 - positive_mask

        # Exclude self-comparisons
        identity_mask = torch.eye(batch_size, device=device)
        positive_mask = positive_mask - identity_mask
        negative_mask = negative_mask * (1.0 - identity_mask)

        # For each anchor, find hardest positive and hardest negative
        # Hardest positive: maximum distance among positives
        positive_distances = distance_matrix * positive_mask
        positive_distances = positive_distances + (1.0 - positive_mask) * -1e9
        hardest_positive, _ = positive_distances.max(dim=1)

        # Hardest negative: minimum distance among negatives
        negative_distances = distance_matrix * negative_mask
        negative_distances = negative_distances + (1.0 - negative_mask) * 1e9
        hardest_negative, _ = negative_distances.min(dim=1)

        # Compute triplet loss
        losses = F.relu(hardest_positive - hardest_negative + self.margin)

        # Only include anchors that have at least one positive
        valid_anchors = positive_mask.sum(dim=1) > 0
        loss = losses[valid_anchors].mean() if valid_anchors.sum() > 0 else torch.tensor(0.0, device=device)

        metrics = {
            "loss": loss.item(),
            "num_valid_anchors": valid_anchors.sum().item(),
            "avg_positive_distance": hardest_positive[valid_anchors].mean().item() if valid_anchors.sum() > 0 else 0.0,
            "avg_negative_distance": hardest_negative[valid_anchors].mean().item() if valid_anchors.sum() > 0 else 0.0,
        }

        return loss, metrics
