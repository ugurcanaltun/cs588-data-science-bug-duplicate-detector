"""
Supervised contrastive loss for duplicate bug detection.

This module implements the contrastive learning objective that encourages
embeddings of duplicate bug reports to be close together while pushing
non-duplicates apart.
"""

import torch
import torch.nn as nn
from typing import Tuple


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
