"""
SBERT-based model for duplicate bug detection.

This module implements a Sentence-BERT encoder that maps augmented bug reports
to fixed-dimensional embedding vectors.
"""

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from typing import List

class BugReportEncoderJina(nn.Module):
    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v3",
        freeze: bool = False,
    ):
        super().__init__()
        # Load with remote code support
        self.encoder = SentenceTransformer(model_name, trust_remote_code=True)
        self.encoder.max_seq_length = 1024 

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
        else:
            # --- CRITICAL FIX FOR JINA V3 ---
            self.encoder.train()
            # Explicitly unfreeze every parameter to ensure the graph stays connected
            for param in self.encoder.parameters():
                param.requires_grad = True
            
            # Jina v3 uses a 'model' attribute for its custom layers
            if hasattr(self.encoder[0], 'auto_model'):
                self.encoder[0].auto_model.train()
            # -------------------------------

        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()

    def forward(self, texts: List[str]) -> torch.Tensor:
        if self.training:
            features = self.encoder.tokenize(texts)
            device = next(self.encoder.parameters()).device
            features = {key: val.to(device) for key, val in features.items()}

            # Manually pass through modules to ensure grad_fn is preserved
            out = features
            for module in self.encoder:
                # Some modules in Jina v3 might require 'task' even in forward
                # If this fails, we can add kwargs here
                out = module(out)
            
            embeddings = out['sentence_embedding']
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            # Final sanity check: if this prints False, the graph is detached in the modeling file
            # print(f"DEBUG: Embeddings require grad: {embeddings.requires_grad}")
            
            return embeddings
        else:
            return self.encoder.encode(
                texts,
                task="text-matching",
                convert_to_tensor=True,
                show_progress_bar=False,
                normalize_embeddings=True
            )
        
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

        # trust_remote_code=True is needed for some models like jina-embeddings-v3
        model.encoder = SentenceTransformer(model_path, trust_remote_code=True)
        model.embedding_dim = model.encoder.get_sentence_embedding_dimension()

        if freeze:
            for param in model.encoder.parameters():
                param.requires_grad = False
            model.encoder.eval()

        return model

class BugReportEncoder(nn.Module):
    """
    SBERT-based encoder for bug reports.

    This encoder takes augmented bug report text and produces a fixed-dimensional
    embedding by extracting the [CLS] token from the last hidden state of a
    pre-trained Sentence-BERT model.

    Args:
        model_name: Name or path of the pre-trained SBERT model
        freeze: If True, freeze the encoder weights (baseline mode)
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        freeze: bool = False,
    ):
        super().__init__()

        # Load pre-trained SBERT model
        # trust_remote_code=True is needed for some models like jina-embeddings-v3
        self.encoder = SentenceTransformer(model_name, trust_remote_code=True)

        self.encoder.max_seq_length = 512  # Set max sequence length

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

        # trust_remote_code=True is needed for some models like jina-embeddings-v3
        model.encoder = SentenceTransformer(model_path, trust_remote_code=True)
        model.embedding_dim = model.encoder.get_sentence_embedding_dimension()

        if freeze:
            for param in model.encoder.parameters():
                param.requires_grad = False
            model.encoder.eval()

        return model
