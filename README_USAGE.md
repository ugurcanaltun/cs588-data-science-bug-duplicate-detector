# SBERT-based Duplicate Bug Detection - Usage Guide

This project implements a contrastive learning approach for detecting duplicate bug reports using pre-trained Sentence-BERT (SBERT) models. The methodology is described in detail in the accompanying research paper.

## Overview

The system supports two experimental approaches:
1. **Approach 1**: VLM outputs + standard bug labels (with screenshot captions)
2. **Approach 2**: Standard bug labels only (text-only baseline)

For each approach, we evaluate:
- **Baseline**: Frozen pre-trained SBERT (no fine-tuning)
- **Proposed**: Fine-tuned SBERT with supervised contrastive learning

## Installation

```bash
# Clone the repository
cd cs588-data-science-bug-duplicate-detector

# Install dependencies
pip install -r requirements.txt
```

## Data Format

Your data should be in JSON or CSV format with the following required fields:

- `bug_id`: Unique identifier for the bug report (int)
- `duplicate_cluster_id`: Cluster ID for grouping duplicates (int)
- `augmented_text_with_vlm`: Full augmented text including VLM-generated sections (string)
- `augmented_text_without_vlm`: Augmented text without VLM sections (string)

See `data_format_example.json` for a complete example.

### Data Structure

```json
[
  {
    "bug_id": 12345,
    "duplicate_cluster_id": 1001,
    "augmented_text_with_vlm": "[PRODUCT] Firefox\n[COMPONENT] ...\n[SCREENSHOT_CAPTION] ...",
    "augmented_text_without_vlm": "[PRODUCT] Firefox\n[COMPONENT] ..."
  },
  ...
]
```

Bug reports with the same `duplicate_cluster_id` are considered duplicates.

## Training

### 1. Baseline (Frozen SBERT)

To establish a baseline using frozen pre-trained SBERT without fine-tuning:

**With VLM (Approach 1):**
```bash
python train.py \
  --train_data data/train.json \
  --val_data data/val.json \
  --use_vlm \
  --freeze \
  --batch_size 64 \
  --epochs 1 \
  --output_dir outputs \
  --experiment_name baseline_frozen_with_vlm
```

**Without VLM (Approach 2):**
```bash
python train.py \
  --train_data data/train.json \
  --val_data data/val.json \
  --freeze \
  --batch_size 64 \
  --epochs 1 \
  --output_dir outputs \
  --experiment_name baseline_frozen_without_vlm
```

### 2. Fine-tuned Model with Contrastive Learning

To fine-tune SBERT with supervised contrastive learning:

**With VLM (Approach 1):**
```bash
python train.py \
  --train_data data/train.json \
  --val_data data/val.json \
  --use_vlm \
  --model_name sentence-transformers/all-MiniLM-L6-v2 \
  --batch_size 32 \
  --samples_per_cluster 4 \
  --epochs 10 \
  --lr 2e-5 \
  --temperature 0.07 \
  --output_dir outputs \
  --experiment_name finetuned_with_vlm \
  --save_best_only
```

**Without VLM (Approach 2):**
```bash
python train.py \
  --train_data data/train.json \
  --val_data data/val.json \
  --model_name sentence-transformers/all-MiniLM-L6-v2 \
  --batch_size 32 \
  --samples_per_cluster 4 \
  --epochs 10 \
  --lr 2e-5 \
  --temperature 0.07 \
  --output_dir outputs \
  --experiment_name finetuned_without_vlm \
  --save_best_only
```

### Training Arguments

Key arguments:

- `--train_data`: Path to training data (JSON or CSV)
- `--val_data`: Path to validation data
- `--use_vlm`: Use VLM-augmented text (Approach 1). Omit for text-only (Approach 2)
- `--freeze`: Freeze encoder weights (baseline mode)
- `--model_name`: Pre-trained SBERT model (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `--batch_size`: Training batch size
- `--samples_per_cluster`: Number of samples per duplicate cluster in each batch (important for contrastive learning)
- `--epochs`: Number of training epochs
- `--lr`: Learning rate
- `--temperature`: Temperature parameter for contrastive loss
- `--save_best_only`: Only save the best checkpoint based on validation Recall@10

For a full list of arguments, run:
```bash
python train.py --help
```

## Evaluation

Evaluate a trained model on the test set:

**Baseline Model:**
```bash
python evaluate.py \
  --model_path sentence-transformers/all-MiniLM-L6-v2 \
  --baseline \
  --test_data data/test.json \
  --use_vlm \
  --batch_size 64 \
  --output_dir evaluation_results
```

**Fine-tuned Model:**
```bash
python evaluate.py \
  --model_path outputs/finetuned_with_vlm/best_model \
  --test_data data/test.json \
  --use_vlm \
  --batch_size 64 \
  --output_dir evaluation_results \
  --save_embeddings
```

### Evaluation Arguments

- `--model_path`: Path to trained model checkpoint (or pre-trained model name for baseline)
- `--baseline`: Flag to indicate baseline evaluation (frozen pre-trained model)
- `--test_data`: Path to test data
- `--use_vlm`: Use VLM-augmented text
- `--k_values`: K values for Recall@k and MAP@k (default: [1, 5, 10, 20])
- `--save_embeddings`: Save test embeddings to disk
- `--save_similarities`: Save similarity matrix to disk

## Metrics

The evaluation script computes the following retrieval metrics:

1. **Recall@k** (k ∈ {1, 5, 10, 20}): Proportion of queries where at least one duplicate appears in top-k results
2. **MRR** (Mean Reciprocal Rank): Average of 1/rank of first relevant item
3. **MAP@k** (Mean Average Precision at k): Rewards systems that rank all duplicates highly

## Output Structure

After training, the output directory will contain:

```
outputs/
└── experiment_name/
    ├── config.json                 # Training configuration
    ├── training_history.json       # Training and validation metrics per epoch
    ├── best_model/                 # Best checkpoint (if --save_best_only)
    │   ├── config.json
    │   ├── model files...
    │   └── training_state.pt
    └── final_model/                # Final model after all epochs
        ├── config.json
        ├── model files...
        └── training_state.pt
```

After evaluation:

```
evaluation_results/
├── metrics_with_vlm.json           # Evaluation metrics
├── metrics_without_vlm.json
├── embeddings_with_vlm.pt         # Test embeddings (if --save_embeddings)
└── similarities_with_vlm.pt       # Similarity matrix (if --save_similarities)
```

## Example Workflow

Here's a complete workflow to replicate the paper's experiments:

```bash
# 1. Baseline with VLM (Approach 1)
python train.py --train_data data/train.json --val_data data/val.json \
  --use_vlm --freeze --epochs 1 --experiment_name baseline_vlm

python evaluate.py --model_path sentence-transformers/all-MiniLM-L6-v2 \
  --baseline --test_data data/test.json --use_vlm

# 2. Baseline without VLM (Approach 2)
python train.py --train_data data/train.json --val_data data/val.json \
  --freeze --epochs 1 --experiment_name baseline_no_vlm

python evaluate.py --model_path sentence-transformers/all-MiniLM-L6-v2 \
  --baseline --test_data data/test.json

# 3. Fine-tuned with VLM (Approach 1)
python train.py --train_data data/train.json --val_data data/val.json \
  --use_vlm --epochs 10 --experiment_name finetuned_vlm --save_best_only

python evaluate.py --model_path outputs/finetuned_vlm/best_model \
  --test_data data/test.json --use_vlm

# 4. Fine-tuned without VLM (Approach 2)
python train.py --train_data data/train.json --val_data data/val.json \
  --epochs 10 --experiment_name finetuned_no_vlm --save_best_only

python evaluate.py --model_path outputs/finetuned_no_vlm/best_model \
  --test_data data/test.json
```

## Model Architecture

The model uses:
- **Encoder**: Pre-trained Sentence-BERT (default: `all-MiniLM-L6-v2`)
- **Pooling**: [CLS] token from last hidden state
- **Normalization**: L2-normalized embeddings for cosine similarity
- **Loss**: Supervised contrastive loss with temperature scaling

## Training Details

### Contrastive Learning

The training uses cluster-balanced batch sampling to ensure each batch contains multiple reports from the same duplicate clusters. This is essential for contrastive learning.

Key parameters:
- `batch_size`: Total samples per batch (default: 32)
- `samples_per_cluster`: Samples from each cluster (default: 4)
- `temperature`: Temperature scaling for similarities (default: 0.07)

### Learning Rate Schedule

- Warmup: Linear warmup for 1 epoch (default)
- Main phase: Cosine annealing decay

### Optimizer

- AdamW with weight decay (default: 0.01)
- Learning rate: 2e-5 (default)

## Advanced Usage

### Custom SBERT Models

You can use any pre-trained Sentence-BERT model from Hugging Face:

```bash
python train.py \
  --model_name sentence-transformers/all-mpnet-base-v2 \
  --train_data data/train.json \
  --val_data data/val.json
```

Popular SBERT models:
- `sentence-transformers/all-MiniLM-L6-v2` (fast, 384-dim)
- `sentence-transformers/all-mpnet-base-v2` (better quality, 768-dim)
- `sentence-transformers/all-roberta-large-v1` (best quality, 1024-dim)

### Alternative Loss Functions

The codebase also includes a Triplet Loss implementation in `src/loss.py` that can be used as an alternative to supervised contrastive loss.

### Using the Models Programmatically

```python
from src.model import BugReportEncoder
from src.metrics import compute_similarity_matrix

# Load trained model
model = BugReportEncoder.load_pretrained('outputs/finetuned_vlm/best_model')

# Encode bug reports
texts = ["[PRODUCT] Firefox\n[SUMMARY] ...", ...]
embeddings = model(texts)

# Compute similarities
similarities = compute_similarity_matrix(embeddings, embeddings)
```

## Troubleshooting

### Out of Memory (OOM)

If you encounter OOM errors:
- Reduce `--batch_size`
- Reduce `--samples_per_cluster`
- Use a smaller SBERT model (e.g., `all-MiniLM-L6-v2`)

### No Valid Anchors in Batch

If you see "No valid anchors" warnings:
- Increase `--batch_size`
- Increase `--samples_per_cluster`
- Check that your data has sufficient duplicate clusters

### Slow Training

- Enable GPU: Ensure `--device cuda` is set
- Increase `--num_workers` for data loading
- Use a smaller SBERT model

## Citation

If you use this code, please cite:

```
[Your paper citation here]
```

## License

[Your license here]
