# Data Preprocessing Guide

This guide explains how to prepare your bug report data for training the duplicate detection model.

## Overview

The preprocessing pipeline takes raw bug report data and creates properly formatted train/dev/test splits with:
- **Augmented text** combining multiple bug report fields in a tagged format
- **Duplicate cluster IDs** for contrastive learning
- **Two versions** of each report: one with VLM outputs, one text-only

## Required Input Files

You need the following files in your `data/` directory:

### 1. Bug Report Data
**File**: `processed_duplicate_training_data.csv`

This CSV file should contain all bug reports with these columns:
- `Issue_id`: Unique bug report ID
- `Title`: Bug report title/summary
- `Description`: Original bug description
- `Enhanced_Description`: Description + VLM-generated outputs (if attachments exist)
- `Priority`: Bug priority (e.g., P1, P2, P3)
- `Component`: Component/module name
- `Product`: Product name (e.g., Firefox)
- `Op_sys`: Operating system
- `attachment`: Attachment information (optional)
- `Duplicated_issue`: Original duplicate field (not used by preprocessing script)

### 2. Train/Dev/Test Split Files
**Files**:
- `data/train/train.csv`
- `data/dev/dev.csv`
- `data/test/test.csv`

Each split CSV file should have:
- `Issue_id`: Bug report ID
- `Duplicate`: Semicolon-separated list of duplicate IDs, or "NULL" for non-duplicates

**Example**:
```csv
Issue_id,Duplicate
320704,NULL
431826,451655
451655,431826
231115,250249;347307;588425;683345
250249,231115;347307;588425;683345
```

## Directory Structure

Before running preprocessing, your data directory should look like:

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

## Running the Preprocessing Script

### Basic Usage

```bash
python preprocess_data.py
```

This will:
1. Load bug reports from `data/processed_duplicate_training_data.csv`
2. Load split information from `data/train/train.csv`, `data/dev/dev.csv`, `data/test/test.csv`
3. Create augmented text for each bug report
4. Build duplicate clusters using Union-Find algorithm
5. Save processed datasets to `data/processed/`

### Custom Paths

```bash
python preprocess_data.py \
    --bug-data path/to/bug_reports.csv \
    --train-split path/to/train.csv \
    --dev-split path/to/dev.csv \
    --test-split path/to/test.csv \
    --output-dir path/to/output
```

### Command-Line Arguments

- `--bug-data`: Path to processed bug report CSV file (default: `data/processed_duplicate_training_data.csv`)
- `--train-split`: Path to train split CSV (default: `data/train/train.csv`)
- `--dev-split`: Path to dev split CSV (default: `data/dev/dev.csv`)
- `--test-split`: Path to test split CSV (default: `data/test/test.csv`)
- `--output-dir`: Output directory for processed files (default: `data/processed`)

## Output Files

After preprocessing, you'll have:

```
data/processed/
├── train.csv
├── train.json
├── dev.csv
├── dev.json
├── test.csv
├── test.json
└── preprocessing_stats.json
```

### Output Format

Each CSV/JSON file contains:
- `bug_id`: Bug report ID
- `duplicate_cluster_id`: Cluster ID (same for all duplicates)
- `augmented_text_with_vlm`: Augmented text using Enhanced_Description
- `augmented_text_without_vlm`: Augmented text using Description only

**Example augmented text**:
```
[PRODUCT] Firefox
[COMPONENT] Bookmarks & History
[OS] Windows 11
[PRIORITY] P3
[SUMMARY] Fullscreen (F11) never shows the bookmarks
[DESCRIPTION] When entering fullscreen mode using F11, the bookmarks toolbar disappears and cannot be shown...
```

### Statistics File

`preprocessing_stats.json` contains:
- Total number of reports per split
- Number of duplicate clusters
- Number of singleton reports (no duplicates)
- Reports in duplicate clusters
- Average and maximum cluster sizes

## Augmented Text Format

The preprocessing script combines bug report fields into a tagged text format:

### Fields Used
1. `[PRODUCT]` - Product name
2. `[COMPONENT]` - Component/module
3. `[OS]` - Operating system
4. `[PRIORITY]` - Bug priority
5. `[SUMMARY]` - Title/summary
6. `[DESCRIPTION]` - Description (or Enhanced_Description for VLM mode)

### Two Versions
- **augmented_text_with_vlm**: Uses `Enhanced_Description` which includes VLM-generated outputs from screenshots
- **augmented_text_without_vlm**: Uses original `Description` only (text-only baseline)

## Duplicate Cluster Assignment

The script uses a **Union-Find algorithm** to build duplicate clusters:

1. Parse all duplicate relationships from split CSV files
2. For each bug report and its duplicates, merge them into the same cluster
3. Assign a unique cluster ID to each group
4. Bug reports without duplicates get their own singleton cluster

**Example**:
```
Issue 123 duplicates: 456, 789
Issue 456 duplicates: 123, 789
Issue 789 duplicates: 123, 456

→ All three get the same cluster_id (e.g., 123)
```

## Verification

After preprocessing, check the statistics:

```bash
cat data/processed/preprocessing_stats.json
```

You should see:
- Reasonable number of duplicate clusters
- Balanced distribution across splits
- Average cluster size > 1 for duplicate clusters

## Next Steps

After preprocessing, you can:

1. **Train the model**:
   ```bash
   python train.py --config config_new_data.json
   ```

2. **Evaluate**:
   ```bash
   python evaluate.py --config config_new_data.json --checkpoint outputs/best_model.pt
   ```

3. **Run all experiments**:
   - Update `run_all_experiments.sh` to use `config_new_data.json`
   - Run the experiments

## Troubleshooting

### Missing Issues

If some issues in the split CSVs are not found in the bug report CSV:
- The script will skip them silently
- Check that `Issue_id` values match between files

### Empty Clusters

If you get errors about empty clusters:
- Check that split CSV files have valid duplicate relationships
- Verify that the `Duplicate` column is properly formatted (semicolon-separated)

### Field Formatting

If you see "NaN" in augmented text:
- Some bug reports may have missing fields (this is normal)
- The script only includes non-empty fields in augmented text

## Example Workflow

```bash
# 1. Prepare your data directory
mkdir -p data/train data/dev data/test

# 2. Place your CSV files in the correct locations
# - data/processed_duplicate_training_data.csv
# - data/train/train.csv
# - data/dev/dev.csv
# - data/test/test.csv

# 3. Run preprocessing
python preprocess_data.py

# 4. Check the output
ls -la data/processed/
cat data/processed/preprocessing_stats.json

# 5. Train the model
python train.py --config config_new_data.json
```

## File Compatibility

The preprocessed files are compatible with the existing training pipeline:
- `src/data.py` already supports CSV and JSON formats
- It automatically selects the correct augmented text column based on the `use_vlm_augmentation` flag
- No changes needed to training/evaluation scripts
