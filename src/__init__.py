"""
SBERT-based Duplicate Bug Detection

This package implements a contrastive learning approach for detecting duplicate bug reports
using pre-trained Sentence-BERT models.
"""

from .model import BugReportEncoder
from .loss import SupervisedContrastiveLoss
from .metrics import RetrievalMetrics, compute_similarity_matrix
from .data import BugReportDataset, create_dataloader, load_data_for_evaluation

__version__ = "1.0.0"

__all__ = [
    "BugReportEncoder",
    "SupervisedContrastiveLoss",
    "RetrievalMetrics",
    "compute_similarity_matrix",
    "BugReportDataset",
    "create_dataloader",
    "load_data_for_evaluation",
]
