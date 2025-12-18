# SBERT with Contrastive Learning for Duplicate Bug Detection

CS 588: Data Science for Software Engineering Term Project

This repository implements a contrastive learning approach for detecting duplicate bug reports using pre-trained Sentence-BERT (SBERT) models. The system learns embeddings where duplicate bug reports are close together in the embedding space, enabling efficient retrieval of similar bugs.

## Overview

Duplicate bug reports waste developer time and effort. This project addresses this by:

1. **Augmenting bug reports** with structured textual fields and VLM-generated descriptions of screenshots
2. **Fine-tuning pre-trained SBERT models** using supervised contrastive learning
3. **Evaluating on retrieval metrics** (Recall@k, MRR, MAP@k) to measure duplicate detection performance

### Key Features

- Pre-trained SBERT encoder with fine-tuning support
- Supervised contrastive loss for learning duplicate embeddings
- Support for VLM-augmented text (screenshots) and text-only baselines
- Comprehensive evaluation metrics (Recall@k, MRR, MAP@k)
- Cluster-balanced batch sampling for effective contrastive learning

## Project Structure

```
.
├── src/
│   ├── __init__.py           # Package initialization
│   ├── model.py              # SBERT-based encoder
│   ├── loss.py               # Supervised contrastive loss
│   ├── metrics.py            # Evaluation metrics (Recall@k, MRR, MAP@k)
│   └── data.py               # Dataset and data loading utilities
├── train.py                  # Training script
├── evaluate.py               # Evaluation script
├── run_baseline.py           # Quick baseline evaluation
├── compare_results.py        # Compare results across experiments
├── run_all_experiments.sh    # Run all experiments
├── requirements.txt          # Python dependencies
├── config_example.json       # Example configuration
├── data_format_example.json  # Example data format
└── README_USAGE.md           # Detailed usage instructions
```

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

Your data should be in JSON or CSV format with these fields:
- `bug_id`: Unique bug report ID
- `duplicate_cluster_id`: Cluster ID grouping duplicates
- `augmented_text_with_vlm`: Full augmented text with VLM outputs
- `augmented_text_without_vlm`: Text-only augmented text

See `data_format_example.json` for details.

### Training

**Fine-tune SBERT with VLM-augmented text:**
```bash
python train.py \
  --train_data data/train.json \
  --val_data data/val.json \
  --use_vlm \
  --epochs 10 \
  --experiment_name finetuned_with_vlm
```

**Fine-tune SBERT with text-only:**
```bash
python train.py \
  --train_data data/train.json \
  --val_data data/val.json \
  --epochs 10 \
  --experiment_name finetuned_without_vlm
```

### Evaluation

**Evaluate baseline (frozen SBERT):**
```bash
python run_baseline.py \
  --test_data data/test.json \
  --use_vlm
```

**Evaluate fine-tuned model:**
```bash
python evaluate.py \
  --model_path outputs/finetuned_with_vlm/best_model \
  --test_data data/test.json \
  --use_vlm
```

### Run All Experiments

```bash
# Run all four experiments:
# 1. Baseline with VLM
# 2. Baseline without VLM
# 3. Fine-tuned with VLM
# 4. Fine-tuned without VLM
./run_all_experiments.sh
```

### Compare Results

```bash
python compare_results.py \
  --output_dir outputs \
  --use_vlm \
  --latex
```

## Methodology

The approach consists of:

1. **Augmented Text Generation**: Combine structured bug fields with VLM-generated screenshot descriptions
2. **SBERT Encoding**: Map augmented text to fixed-dimensional embeddings
3. **Contrastive Learning**: Train with supervised contrastive loss to bring duplicates close together
4. **Retrieval Evaluation**: Rank candidates by cosine similarity and measure retrieval quality

### Loss Function

Supervised contrastive loss for anchor $i$:

$$L_i = -\frac{1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_p))}{\sum_{a \in A(i)} \exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_a))}$$

where $P(i)$ are positive pairs (same cluster) and $A(i)$ are all other samples in the batch.

### Evaluation Metrics

- **Recall@k**: Proportion of queries with at least one duplicate in top-k
- **MRR**: Mean reciprocal rank of first relevant item
- **MAP@k**: Mean average precision at k (rewards ranking all duplicates highly)

## Experiments

We evaluate four configurations:

| Configuration | VLM Augmentation | Fine-tuning |
|--------------|------------------|-------------|
| Baseline (VLM) | Yes | No |
| Baseline (Text-only) | No | No |
| Fine-tuned (VLM) | Yes | Yes |
| Fine-tuned (Text-only) | No | Yes |

## Results

After running experiments, results are saved in:
```
outputs/
├── baseline_with_vlm/
│   └── metrics_with_vlm.json
├── baseline_without_vlm/
│   └── metrics_without_vlm.json
├── finetuned_with_vlm/
│   ├── best_model/
│   ├── training_history.json
│   └── evaluation_results/
│       └── metrics_with_vlm.json
└── finetuned_without_vlm/
    ├── best_model/
    ├── training_history.json
    └── evaluation_results/
        └── metrics_without_vlm.json
```

## Documentation

See [README_USAGE.md](README_USAGE.md) for detailed usage instructions, including:
- Data format specifications
- Training hyperparameters
- Advanced usage and customization
- Troubleshooting

## Requirements

- Python 3.8+
- PyTorch 2.0+
- sentence-transformers 2.2+
- pandas, numpy, tqdm

See `requirements.txt` for complete list.

## License

[Your license here]

## Citation

If you use this code, please cite:

```
[Your paper citation here]
```
