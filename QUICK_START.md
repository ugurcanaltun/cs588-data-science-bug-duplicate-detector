# Quick Start Guide: From Raw Data to Trained Model

This guide walks you through the complete workflow from raw bug report data to a trained duplicate detection model.

## Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt
```

## Step 1: Prepare Your Data

Create the following directory structure:

```
data/
├── processed_duplicate_training_data.csv
├── train/
│   └── train.csv
├── dev/
│   └── dev.csv
└── test/
    └── test.csv
```

### File Requirements

**processed_duplicate_training_data.csv** must have:
- Issue_id
- Title
- Description
- Enhanced_Description (Description + VLM outputs)
- Priority
- Component
- Product
- Op_sys

**train.csv, dev.csv, test.csv** must have:
- Issue_id
- Duplicate (semicolon-separated IDs or "NULL")

## Step 2: Preprocess the Data

```bash
python preprocess_data.py
```

This will:
- Load all bug reports
- Create augmented text combining fields in tagged format
- Build duplicate clusters using Union-Find
- Save processed files to `data/processed/`

### Expected Output

```
data/processed/
├── train.csv         # Ready for training
├── train.json        # Same data in JSON format
├── dev.csv           # Ready for validation
├── dev.json
├── test.csv          # Ready for testing
├── test.json
└── preprocessing_stats.json  # Statistics
```

### Verify Preprocessing

```bash
# Check statistics
cat data/processed/preprocessing_stats.json

# View first few rows of processed data
head -20 data/processed/train.csv
```

## Step 3: Configure Your Experiment

Use the provided configuration file or create your own:

**config_new_data.json** (already created):
```json
{
  "data": {
    "train_data": "data/processed/train.csv",
    "val_data": "data/processed/dev.csv",
    "test_data": "data/processed/test.csv",
    "use_vlm_augmentation": true
  },
  "training": {
    "batch_size": 32,
    "epochs": 10,
    "learning_rate": 2e-5
  }
}
```

**For text-only baseline**, change `use_vlm_augmentation` to `false`.

## Step 4: Train the Model

### Option A: Using Config File (Recommended)

```bash
# Train with VLM augmentation
python train.py --config config_new_data.json
```

### Option B: Using Command-Line Arguments

```bash
python train.py \
  --train_data data/processed/train.csv \
  --val_data data/processed/dev.csv \
  --use_vlm \
  --epochs 10 \
  --batch_size 32 \
  --learning_rate 2e-5 \
  --experiment_name my_experiment
```

### Training Output

The model will save:
```
outputs/
└── my_experiment/
    ├── best_model.pt          # Best checkpoint
    ├── final_model.pt         # Final checkpoint
    ├── training_history.json  # Loss and metrics per epoch
    └── config.json            # Configuration used
```

## Step 5: Evaluate the Model

### Evaluate Baseline (No Training)

```bash
python run_baseline.py \
  --test_data data/processed/test.csv \
  --use_vlm \
  --output_dir outputs/baseline_with_vlm
```

### Evaluate Fine-tuned Model

```bash
python evaluate.py \
  --model_path outputs/my_experiment/best_model.pt \
  --test_data data/processed/test.csv \
  --use_vlm \
  --output_dir outputs/my_experiment/evaluation
```

### Results

Metrics will be saved in `metrics.json`:
```json
{
  "recall@1": 0.65,
  "recall@5": 0.82,
  "recall@10": 0.88,
  "recall@20": 0.92,
  "mrr": 0.73,
  "map@1": 0.65,
  "map@5": 0.71,
  "map@10": 0.73,
  "map@20": 0.74
}
```

## Step 6: Compare Experiments

Run multiple experiments and compare:

```bash
# Train baseline with VLM
python run_baseline.py --test_data data/processed/test.csv --use_vlm

# Train baseline without VLM
python run_baseline.py --test_data data/processed/test.csv

# Train fine-tuned with VLM
python train.py --config config_new_data.json

# Train fine-tuned without VLM (modify config first)
python train.py --config config_new_data.json

# Compare results
python compare_results.py --output_dir outputs
```

## Complete Workflow Example

```bash
# 1. Preprocess data
python preprocess_data.py

# 2. Check preprocessing succeeded
cat data/processed/preprocessing_stats.json

# 3. Train model
python train.py --config config_new_data.json

# 4. Evaluate
python evaluate.py \
  --model_path outputs/sbert_contrastive_with_vlm/best_model.pt \
  --test_data data/processed/test.csv \
  --use_vlm

# 5. View results
cat outputs/sbert_contrastive_with_vlm/evaluation/metrics.json
```

## Troubleshooting

### "No such file or directory: data/processed_duplicate_training_data.csv"

Make sure you have:
1. Created the `data/` directory
2. Placed your CSV file with the exact name `processed_duplicate_training_data.csv`

### "Missing required column: ..."

Check that your CSV files have the required columns. See Step 1.

### "No clusters have at least 4 samples"

Your dataset may have too many singleton clusters. Try:
```bash
python train.py --config config_new_data.json --samples_per_cluster 2
```

### Low Recall Scores

This could mean:
- Not enough training epochs
- Dataset imbalance (too many singletons)
- Hyperparameters need tuning

Try:
- Increasing epochs to 15-20
- Reducing learning rate to 1e-5
- Adjusting temperature (try 0.05 or 0.1)

## Next Steps

- **Experiment with hyperparameters**: Try different learning rates, batch sizes, temperatures
- **Analyze errors**: Look at which bug pairs are not being retrieved
- **Try different base models**: Experiment with larger SBERT models
- **Ablation studies**: Test impact of different bug report fields

## Additional Resources

- **[DATA_PREPROCESSING.md](DATA_PREPROCESSING.md)** - Detailed preprocessing guide
- **[README.md](README.md)** - Full project documentation
- **[README_USAGE.md](README_USAGE.md)** - Advanced usage instructions
