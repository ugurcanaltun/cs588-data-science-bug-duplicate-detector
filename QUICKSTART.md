# Quick Start Guide

This guide will help you get started with the SBERT-based duplicate bug detection system in under 5 minutes.

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended but not required)
- Your augmented bug report data in JSON or CSV format

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install PyTorch, sentence-transformers, and other required packages.

## Step 2: Prepare Your Data

Ensure your data files are in the correct format. Each file should contain:

**Required fields:**
- `bug_id`: Unique identifier (int)
- `duplicate_cluster_id`: Cluster ID for grouping duplicates (int)
- `augmented_text_with_vlm`: Full text including VLM outputs (string)
- `augmented_text_without_vlm`: Text-only version (string)

**Example structure:**
```json
[
  {
    "bug_id": 12345,
    "duplicate_cluster_id": 1001,
    "augmented_text_with_vlm": "[PRODUCT] Firefox\n[SUMMARY] ...\n[SCREENSHOT_CAPTION] ...",
    "augmented_text_without_vlm": "[PRODUCT] Firefox\n[SUMMARY] ..."
  },
  ...
]
```

See `data_format_example.json` for a complete example.

Place your data files in a `data/` directory:
```
data/
├── train.json
├── val.json
└── test.json
```

## Step 3: Run a Quick Test

Test that everything works with the baseline model (no training required):

```bash
python run_baseline.py \
  --test_data data/test.json \
  --use_vlm \
  --output_dir test_output
```

This should complete in a few minutes and produce evaluation metrics.

## Step 4: Train Your First Model

Train a fine-tuned model with VLM augmentation:

```bash
python train.py \
  --train_data data/train.json \
  --val_data data/val.json \
  --use_vlm \
  --epochs 5 \
  --batch_size 32 \
  --experiment_name my_first_model
```

Training time depends on your dataset size and hardware. With a GPU:
- Small dataset (1k reports): ~5-10 minutes
- Medium dataset (10k reports): ~30-60 minutes
- Large dataset (100k reports): ~3-5 hours

## Step 5: Evaluate Your Model

Evaluate the trained model on the test set:

```bash
python evaluate.py \
  --model_path outputs/my_first_model/best_model \
  --test_data data/test.json \
  --use_vlm \
  --save_embeddings
```

## Step 6: View Results

Results are saved in JSON format:

```bash
cat outputs/my_first_model/evaluation_results/metrics_with_vlm.json
```

You should see metrics like:
```json
{
  "recall@1": 0.4523,
  "recall@5": 0.7234,
  "recall@10": 0.8156,
  "recall@20": 0.8892,
  "mrr": 0.5891,
  "map@1": 0.4523,
  "map@5": 0.5634,
  "map@10": 0.5923,
  "map@20": 0.6012,
  "num_queries": 1523
}
```

## Next Steps

### Run All Experiments

To replicate the full paper experiments:

```bash
./run_all_experiments.sh
```

This will run all four configurations:
1. Baseline with VLM
2. Baseline without VLM
3. Fine-tuned with VLM
4. Fine-tuned without VLM

### Compare Results

Compare metrics across all experiments:

```bash
python compare_results.py \
  --output_dir outputs \
  --use_vlm \
  --save_csv comparison.csv \
  --latex
```

This generates a comparison table and LaTeX format for your paper.

## Common Issues and Solutions

### Issue: Out of Memory (OOM)

**Solution**: Reduce batch size
```bash
python train.py ... --batch_size 16
```

Or use a smaller model:
```bash
python train.py ... --model_name sentence-transformers/all-MiniLM-L6-v2
```

### Issue: No valid anchors in batch

**Solution**: Increase batch size or samples per cluster
```bash
python train.py ... --batch_size 64 --samples_per_cluster 4
```

### Issue: Training is slow

**Solutions**:
1. Enable GPU: `--device cuda`
2. Increase data loading workers: `--num_workers 4`
3. Use a smaller model: `--model_name sentence-transformers/all-MiniLM-L6-v2`

### Issue: Data loading error

**Solution**: Verify your data format matches the example in `data_format_example.json`

```bash
# Test data loading
python -c "from src.data import BugReportDataset; ds = BugReportDataset('data/train.json'); print(f'Loaded {len(ds)} reports')"
```

## Tips for Best Results

1. **Data Quality**: Ensure augmented texts are well-formatted with proper field tags
2. **Batch Sampling**: Use at least 4 samples per cluster for effective contrastive learning
3. **Training Duration**: Train for at least 10 epochs for convergence
4. **Validation**: Monitor validation metrics to avoid overfitting
5. **Multiple Runs**: Run experiments with different random seeds for robust results

## Hardware Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 16 GB
- Storage: 10 GB

**Recommended:**
- CPU: 8+ cores
- RAM: 32 GB
- GPU: NVIDIA GPU with 8+ GB VRAM (e.g., RTX 3070, V100)
- Storage: 50 GB

## Expected Performance

With the default `all-MiniLM-L6-v2` model and Mozilla/Firefox dataset:

**Baseline (frozen):**
- Recall@10: ~0.40-0.50
- MRR: ~0.30-0.40

**Fine-tuned:**
- Recall@10: ~0.70-0.80
- MRR: ~0.55-0.65

Your results may vary depending on:
- Dataset characteristics
- Data quality
- Hyperparameter choices

## Getting Help

1. Check `README_USAGE.md` for detailed documentation
2. Review `IMPLEMENTATION_SUMMARY.md` for technical details
3. Look at example configurations in `config_example.json`
4. Examine example data format in `data_format_example.json`

## Checklist

Before running experiments, ensure:

- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Data files prepared in correct format
- [ ] Data files placed in `data/` directory
- [ ] Baseline test completed successfully
- [ ] GPU available (if using CUDA)
- [ ] Sufficient disk space for outputs

## Time Estimates

For a dataset with ~10,000 bug reports:

| Task | Time (GPU) | Time (CPU) |
|------|-----------|-----------|
| Install dependencies | 2-5 min | 2-5 min |
| Baseline evaluation | 5 min | 15 min |
| Train 1 model (10 epochs) | 30 min | 3-4 hours |
| Evaluate 1 model | 5 min | 15 min |
| All 4 experiments | 2 hours | 12-16 hours |

## Success Indicators

You'll know everything is working correctly when:

1. Baseline evaluation completes without errors
2. Training loss decreases over epochs
3. Validation Recall@10 improves during training
4. Evaluation metrics are computed successfully
5. Fine-tuned model outperforms baseline

## Ready to Start?

Run this command to verify your setup:

```bash
# Test installation
python -c "import torch; import sentence_transformers; print('Setup OK!')"

# Test data loading (replace with your data path)
python -c "from src.data import BugReportDataset; ds = BugReportDataset('data/train.json'); print(f'Data OK! Loaded {len(ds)} reports')"
```

If both commands succeed, you're ready to run your experiments!

```bash
# Start with baseline
python run_baseline.py --test_data data/test.json --use_vlm

# Then train your model
python train.py --train_data data/train.json --val_data data/val.json --use_vlm --epochs 10
```

Good luck with your experiments!
