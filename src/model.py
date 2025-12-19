"""
SBERT-based model for duplicate bug detection.

This module implements a Sentence-BERT encoder that maps augmented bug reports
to fixed-dimensional embedding vectors.
"""

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from typing import List


class BugReportEncoder(nn.Module):
    """
    SBERT-based encoder for bug reports.

    This encoder takes augmented bug report text and produces a fixed-dimensional
    embedding by extracting the [CLS] token from the last hidden state of a
    pre-trained Sentence-BERT model.

    Args:
        model_name: Name or path of the pre-trained SBERT model
        freeze: If True, freeze the encoder weights (baseline mode)
        max_length: Maximum sequence length for tokenization
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        freeze: bool = False,
        max_length: int = 512
    ):
        super().__init__()

        # Load pre-trained SBERT model
        self.encoder = SentenceTransformer(model_name)
        self.max_length = max_length

        # Freeze parameters if baseline mode
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

        # Get embedding dimension
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()

    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Encode bug reports into embeddings.

        Args:
            texts: List of augmented bug report texts

        Returns:
            Tensor of shape (batch_size, embedding_dim) containing L2-normalized embeddings
        """
        # If in training mode, use the model directly to maintain gradients
        if self.training:
            # Tokenize the texts
            features = self.encoder.tokenize(texts)

            # Move features to the same device as the model
            device = next(self.encoder.parameters()).device
            features = {key: val.to(device) for key, val in features.items()}

            # Forward pass through the SentenceTransformer model
            embeddings = self.encoder(features)['sentence_embedding']

            # L2 normalize for cosine similarity
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        else:
            # In eval mode, use the optimized encode method
            embeddings = self.encoder.encode(
                texts,
                convert_to_tensor=True,
                show_progress_bar=False,
                normalize_embeddings=True
            )

        return embeddings

    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> torch.Tensor:
        """
        Encode a large batch of texts with mini-batching for efficiency.

        Args:
            texts: List of augmented bug report texts
            batch_size: Size of mini-batches for encoding
            show_progress: Whether to show progress bar

        Returns:
            Tensor of shape (num_texts, embedding_dim) containing L2-normalized embeddings
        """
        embeddings = self.encoder.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=show_progress,
            normalize_embeddings=True
        )

        return embeddings

    def save_pretrained(self, save_path: str):
        """Save the model to disk."""
        self.encoder.save(save_path)

    @classmethod
    def load_pretrained(cls, model_path: str, freeze: bool = False) -> "BugReportEncoder":
        """
        Load a saved model from disk.

        Args:
            model_path: Path to the saved model
            freeze: Whether to freeze the loaded model

        Returns:
            Loaded BugReportEncoder instance
        """
        model = cls.__new__(cls)
        super(BugReportEncoder, model).__init__()

        model.encoder = SentenceTransformer(model_path)
        model.embedding_dim = model.encoder.get_sentence_embedding_dimension()
        model.max_length = 512

        if freeze:
            for param in model.encoder.parameters():
                param.requires_grad = False
            model.encoder.eval()

        return model

