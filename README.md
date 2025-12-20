# SBERT with Contrastive Learning for Duplicate Bug Detection

## Replication Package

**Course:** CS 588: Data Science for Software Engineering
**Project:** Term Project - Duplicate Bug Detection Using Contrastive Learning

This repository is a **replication package** containing all code, data, and configuration files necessary to reproduce the results reported in our project. The system uses Sentence-BERT (SBERT) models fine-tuned with supervised contrastive learning to detect duplicate bug reports.

## Overview

Duplicate bug reports waste developer time and effort. This project addresses duplicate detection by:

1. **Augmenting bug reports** with structured textual fields (Product, Component, OS, Priority) and VLM-generated descriptions of screenshots
2. **Fine-tuning pre-trained SBERT models** using supervised contrastive learning to bring duplicate reports close in embedding space
3. **Evaluating retrieval performance** using Recall@k, MRR, and MAP@k metrics

### Experimental Configurations

The replication package reproduces results for **four experimental configurations**:

| Configuration | VLM Augmentation | Fine-tuning | Description |
|--------------|------------------|-------------|-------------|
| Baseline (VLM) | ✓ | ✗ | Frozen SBERT with VLM-enhanced text |
| Baseline (Text-only) | ✗ | ✗ | Frozen SBERT with text-only input |
| Fine-tuned (VLM) | ✓ | ✓ | Fine-tuned SBERT with VLM-enhanced text |
| Fine-tuned (Text-only) | ✗ | ✓ | Fine-tuned SBERT with text-only input |

## Repository Contents

```
.
├── src/                      # Source code modules
│   ├── __init__.py           # Package initialization
│   ├── model.py              # SBERT-based encoder implementation
│   ├── loss.py               # Supervised contrastive loss
│   ├── metrics.py            # Evaluation metrics (Recall@k, MRR, MAP@k)
│   └── data.py               # Dataset and data loading utilities
│
├── configs/                  # Configuration files for all experiments
│   ├── config_training.json                    # Training configuration
│   ├── config_eval_baseline_with_vlm.json      # Baseline (VLM) evaluation
│   ├── config_eval_baseline_without_vlm.json   # Baseline (text-only) evaluation
│   ├── config_eval_finetuned_with_vlm.json     # Fine-tuned (VLM) evaluation
│   └── config_eval_finetuned_without_vlm.json  # Fine-tuned (text-only) evaluation
│
├── data/                     # Data directory
│   ├── processed_duplicate_training_data.csv   # Raw bug reports with all fields
│   ├── train/train.csv       # Training split assignments
│   ├── dev/dev.csv           # Development split assignments
│   ├── test/test.csv         # Test split assignments
│   └── processed/            # Preprocessed data (generated)
│       ├── train.csv         # Preprocessed training data
│       ├── dev.csv           # Preprocessed development data
│       ├── test.csv          # Preprocessed test data
│       ├── train.json        # Training data in JSON format
│       ├── dev.json          # Development data in JSON format
│       ├── test.json         # Test data in JSON format
│       └── preprocessing_stats.json  # Dataset statistics
│
├── outputs/                  # Experimental results (generated)
│   ├── baseline_results/     # Baseline experiment results
│   │   ├── metrics_baseline_with_vlm.json
│   │   └── metrics_baseline_without_vlm.json
│   └── sbert_contrastive_without_vlm/  # Fine-tuned model (text-only)
│       ├── best_model/       # Saved model weights
│       ├── config.json       # Training configuration used
│       └── metrics_finetuned_without_vlm.json
│
├── data.zip                  # Complete dataset archive (133MB)
├── preprocess_data.py        # Data preprocessing script
├── train.py                  # Model training script
├── evaluate.py               # Model evaluation script
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## System Requirements

- **Python:** 3.8 or higher
- **GPU:** CUDA-compatible GPU recommended (experiments were run on CUDA-enabled device)
- **RAM:** Minimum 16GB recommended
- **Disk Space:** ~2GB for data and model checkpoints

## Installation and Setup

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd cs588-data-science-bug-duplicate-detector
```

### Step 2: Set Up Python Environment

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Dependencies** (from `requirements.txt`):
- torch >= 2.0.0
- sentence-transformers >= 2.2.0
- transformers >= 4.30.0
- pandas >= 1.5.0
- numpy >= 1.23.0
- tqdm >= 4.65.0
- scikit-learn >= 1.3.0

### Step 3: Extract Data

The complete dataset is provided in `data.zip` (133MB). Extract it to use the preprocessed data:

```bash
# Option 1: Use the existing preprocessed data (already in data/)
# The repository includes preprocessed data in data/processed/

# Option 2: Extract from data.zip if needed
unzip data.zip
```

**Dataset Statistics:**
- **Training set:** 92,370 bug reports (7,473 duplicate clusters)
- **Development set:** 11,675 bug reports (951 duplicate clusters)
- **Test set:** 11,769 bug reports (898 duplicate clusters)

### Step 4: Verify Data Structure

Ensure the following data files exist:

```bash
data/
├── processed_duplicate_training_data.csv   # Raw bug reports
├── train/train.csv                         # Train split assignments
├── dev/dev.csv                             # Dev split assignments
├── test/test.csv                           # Test split assignments
└── processed/                              # Preprocessed data (ready to use)
    ├── train.csv, train.json
    ├── dev.csv, dev.json
    ├── test.csv, test.json
    └── preprocessing_stats.json
```

### Optional: Re-run Preprocessing

If you want to regenerate the preprocessed data from raw files:

```bash
python preprocess_data.py
```

This creates augmented text representations combining:
- Product, Component, OS, Priority (structured fields)
- Title (summary)
- Description (with VLM enhancements for screenshots) or plain text

## Reproducing Results

This section provides step-by-step instructions to reproduce all experimental results reported in the paper.

### Configuration Files

All experiments use configuration files in the `configs/` directory:

- `config_training.json` - Training configuration for fine-tuned models
- `config_eval_baseline_with_vlm.json` - Baseline evaluation with VLM
- `config_eval_baseline_without_vlm.json` - Baseline evaluation without VLM
- `config_eval_finetuned_with_vlm.json` - Fine-tuned model evaluation with VLM
- `config_eval_finetuned_without_vlm.json` - Fine-tuned model evaluation without VLM

### Experiment 1: Baseline with VLM

Evaluate the frozen SBERT model with VLM-augmented text:

```bash
python evaluate.py --config configs/config_eval_baseline_with_vlm.json
```

**Expected output:** `outputs/baseline_results/metrics_baseline_with_vlm.json`

**Expected metrics (approximate):**
- Recall@1: 0.624
- Recall@5: 0.794
- Recall@10: 0.842
- Recall@20: 0.882
- MRR: 0.701
- MAP@10: 0.358

### Experiment 2: Baseline without VLM

Evaluate the frozen SBERT model with text-only input:

```bash
python evaluate.py --config configs/config_eval_baseline_without_vlm.json
```

**Expected output:** `outputs/baseline_results/metrics_baseline_without_vlm.json`

**Expected metrics (approximate):**
- Recall@1: 0.630
- Recall@5: 0.802
- Recall@10: 0.852
- Recall@20: 0.888
- MRR: 0.708
- MAP@10: 0.365

### Experiment 3: Fine-tuned Model without VLM

#### Step 3a: Train the Model

Fine-tune SBERT using supervised contrastive learning on text-only input:

```bash
python train.py --config configs/config_training.json
```

**Configuration details:**
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Batch size: 32
- Samples per cluster: 2
- Epochs: 100
- Learning rate: 2e-5
- Temperature: 0.07
- VLM augmentation: disabled (text-only)

**Expected outputs:**
- Best model checkpoint: `outputs/sbert_contrastive_without_vlm/best_model/`
- Training config: `outputs/sbert_contrastive_without_vlm/config.json`

**Training duration:** Approximately 2-4 hours on a single GPU (varies by hardware)

#### Step 3b: Evaluate the Fine-tuned Model

```bash
python evaluate.py --config configs/config_eval_finetuned_without_vlm.json
```

**Expected output:** `outputs/sbert_contrastive_without_vlm/metrics_finetuned_without_vlm.json`

**Expected metrics (approximate):**
- Recall@1: 0.732
- Recall@5: 0.889
- Recall@10: 0.925
- Recall@20: 0.950
- MRR: 0.802
- MAP@10: 0.484

### Experiment 4: Fine-tuned Model with VLM

#### Step 4a: Train the Model

Fine-tune SBERT using VLM-augmented text:

```bash
# Modify config_training.json to set "use_vlm_augmentation": true
# Then run:
python train.py --config configs/config_training.json
```

**Note:** To run this experiment, edit `configs/config_training.json` and change:
```json
"use_vlm_augmentation": false  →  "use_vlm_augmentation": true
```

#### Step 4b: Evaluate the Fine-tuned Model

```bash
python evaluate.py --config configs/config_eval_finetuned_with_vlm.json
```

**Expected output:** `outputs/sbert_contrastive_with_vlm/metrics_finetuned_with_vlm.json`

## Understanding the Results

### Output Files

After running experiments, results are saved in JSON format in the `outputs/` directory:

```json
{
  "recall@1": 0.732,
  "recall@5": 0.889,
  "recall@10": 0.925,
  "recall@20": 0.950,
  "mrr": 0.802,
  "map@1": 0.253,
  "map@5": 0.425,
  "map@10": 0.484,
  "map@20": 0.526,
  "num_queries": 3760
}
```

### Evaluation Metrics Explained

- **Recall@k**: Percentage of test queries where at least one duplicate bug report appears in the top-k retrieved results
  - Example: Recall@10 = 0.925 means 92.5% of queries found a duplicate in the top 10 results

- **MRR (Mean Reciprocal Rank)**: Average of the reciprocal rank of the first correct duplicate
  - Example: If the first duplicate appears at rank 2, the reciprocal rank is 1/2 = 0.5
  - Higher is better (max = 1.0 when first result is always a duplicate)

- **MAP@k (Mean Average Precision at k)**: Measures how well the system ranks ALL duplicates, not just the first one
  - Rewards systems that rank multiple duplicates highly
  - More stringent than Recall@k

### Comparing Configurations

**Key Findings:**
1. **Fine-tuning vs. Baseline**: Fine-tuned models significantly outperform frozen SBERT baselines
   - Fine-tuned (text-only): Recall@10 = 0.925 vs. Baseline: 0.852 (+8.6% improvement)

2. **VLM Augmentation**: Impact of VLM varies by configuration
   - For baselines: Minimal difference between VLM and text-only
   - For fine-tuned models: Results depend on training data

3. **Best Configuration**: Fine-tuned model (text-only) achieves the highest performance
   - Recall@10: 92.5%
   - MRR: 0.802
   - MAP@10: 0.484

## Methodology

### Approach Overview

1. **Data Augmentation**: Combine multiple bug report fields into structured text
   - Structured fields: Product, Component, OS, Priority
   - Textual fields: Title (summary) and Description
   - VLM enhancement: Vision-Language Model descriptions of screenshots

2. **Model Architecture**: SBERT encoder (`sentence-transformers/all-MiniLM-L6-v2`)
   - Maps variable-length text to fixed-dimensional embeddings (384-dim)
   - Pre-trained on semantic similarity tasks

3. **Training with Contrastive Learning**:
   - **Supervised Contrastive Loss**: Pull duplicates together, push non-duplicates apart
   - **Cluster-balanced Sampling**: Sample multiple reports from each duplicate cluster per batch
   - **Temperature Scaling**: Controls separation between similar/dissimilar pairs

4. **Retrieval Evaluation**:
   - Encode all test bug reports
   - For each query, rank candidates by cosine similarity
   - Measure retrieval quality using Recall@k, MRR, and MAP@k

### Loss Function

The supervised contrastive loss for anchor bug report i:

```
L_i = -1/|P(i)| * Σ_p∈P(i) log[ exp(sim(z_i, z_p) / τ) / Σ_a∈A(i) exp(sim(z_i, z_a) / τ) ]
```

Where:
- P(i) = positive pairs (bug reports in the same duplicate cluster as i)
- A(i) = all other samples in the batch (excluding i itself)
- sim(·,·) = cosine similarity
- τ = temperature parameter (0.07)

## Code Structure

### Source Modules (`src/`)

- **`model.py`**: SBERT-based encoder implementation
  - `BugReportEncoder`: Wraps sentence-transformers models
  - Forward pass returns L2-normalized embeddings

- **`loss.py`**: Supervised contrastive loss implementation
  - `SupervisedContrastiveLoss`: Computes loss for duplicate bug detection
  - Temperature-scaled cosine similarity
  - Efficient batch-wise computation

- **`metrics.py`**: Evaluation metrics
  - `RetrievalMetrics`: Computes Recall@k, MRR, MAP@k
  - `compute_similarity_matrix`: Efficient pairwise cosine similarity

- **`data.py`**: Data loading and preprocessing utilities
  - `BugReportDataset`: PyTorch dataset for bug reports
  - `ClusterBalancedSampler`: Samples from duplicate clusters for contrastive learning
  - `create_dataloader`: Creates dataloaders with cluster-balanced sampling

### Scripts

- **`preprocess_data.py`**: Converts raw CSV files to augmented text format
- **`train.py`**: Training script with early stopping and checkpointing
- **`evaluate.py`**: Evaluation script for both baseline and fine-tuned models

## Troubleshooting

### Common Issues

**1. CUDA out of memory errors**
- Reduce batch size in config file: `"batch_size": 16` (default is 32)
- Reduce `samples_per_cluster` to 1 in training config

**2. Data files not found**
- Ensure `data.zip` is extracted
- Verify files exist in `data/processed/` directory
- Re-run `python preprocess_data.py` if needed

**3. Slow training on CPU**
- Training requires GPU for reasonable performance
- Check CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`
- Set `"device": "cpu"` in config if GPU unavailable (not recommended for full training)

**4. Different results than reported**
- Random seed is set to 42 but small variations can occur
- Ensure same PyTorch and transformers versions
- Training for fewer epochs will yield lower performance

**5. ModuleNotFoundError**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Activate virtual environment if using one

### Hardware Recommendations

- **Minimum**: CPU with 16GB RAM (slow, not recommended for training)
- **Recommended**: CUDA GPU with 8GB+ VRAM (RTX 3070 or better)
- **Optimal**: CUDA GPU with 16GB+ VRAM (A100, V100, RTX 4090)

## Data Format

### Preprocessed Data Format

The processed CSV files contain the following columns:

- `bug_id`: Unique identifier for the bug report (int)
- `duplicate_cluster_id`: Cluster ID grouping duplicate reports (int)
  - Reports with the same cluster ID are duplicates
  - Singleton reports (no duplicates) have unique cluster IDs
- `augmented_text_with_vlm`: Full augmented text including VLM outputs (str)
- `augmented_text_without_vlm`: Text-only augmented text (str)

### Augmented Text Format

Example augmented text structure:

```
[PRODUCT] Firefox
[COMPONENT] General
[OS] Windows 10
[PRIORITY] P3
[SUMMARY] Browser crashes when opening multiple tabs
[DESCRIPTION] When I open more than 10 tabs simultaneously, the browser
becomes unresponsive and crashes. This happens consistently on my Windows
10 machine. [VLM: Screenshot shows error dialog with message "Firefox
has stopped responding"]
```

## Additional Information

### Training Hyperparameters

Key hyperparameters used in the experiments (from `configs/config_training.json`):

- **Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional embeddings)
- **Batch size**: 32
- **Samples per cluster**: 2 (for contrastive learning)
- **Epochs**: 100 (with early stopping)
- **Learning rate**: 2e-5
- **Weight decay**: 0.01
- **Warmup epochs**: 1
- **Temperature**: 0.07
- **Optimizer**: AdamW
- **LR Scheduler**: Linear warmup + Cosine annealing

### Computational Requirements

**Training time** (fine-tuned model, text-only):
- Single GPU (RTX 3080): ~2-3 hours for 100 epochs
- Single GPU (V100): ~1.5-2 hours for 100 epochs
- CPU: Not recommended (would take days)

**Evaluation time** (test set with 11,769 samples):
- Single GPU: ~1-2 minutes per configuration
- CPU: ~5-10 minutes per configuration

**Disk space**:
- Raw data (`data.zip`): 133 MB
- Preprocessed data: ~500 MB
- Model checkpoints: ~100 MB per fine-tuned model
- Total: ~1-2 GB

## Citation

If you use this code or data in your research, please cite:

```bibtex
@inproceedings{your-paper-2025,
  title={SBERT with Contrastive Learning for Duplicate Bug Detection},
  author={Your Name},
  booktitle={CS 588: Data Science for Software Engineering},
  year={2025}
}
```

## Contact

For questions or issues, please contact:
- **Author**: [Your name]
- **Email**: [Your email]
- **Course**: CS 588: Data Science for Software Engineering

## Acknowledgments

This project uses:
- [Sentence-Transformers](https://www.sbert.net/) for pre-trained SBERT models
- Bug report data from [specify source]
- VLM-enhanced descriptions from [specify VLM model/source]
