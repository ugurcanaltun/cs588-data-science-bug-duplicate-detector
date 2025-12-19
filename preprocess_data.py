"""
Preprocess bug report data and create train/dev/test splits.

This script:
1. Loads the processed_duplicate_training_data.csv with bug report fields
2. Loads train/dev/test split information from CSV files
3. Creates augmented text combining multiple fields in tagged format
4. Assigns duplicate cluster IDs based on the split information
5. Saves processed datasets ready for model training
"""

import pandas as pd
import json
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict


def create_augmented_text(row: pd.Series, use_vlm: bool = True) -> str:
    """
    Create augmented text from bug report fields.

    Args:
        row: DataFrame row containing bug report fields
        use_vlm: If True, use Enhanced_Description (includes VLM outputs).
                 If False, use Description only (text-only baseline).

    Returns:
        Augmented text string with tagged fields
    """
    sections = []

    # Add PRODUCT field
    if pd.notna(row['Product']) and str(row['Product']).strip():
        sections.append(f"[PRODUCT] {row['Product']}")

    # Add COMPONENT field
    if pd.notna(row['Component']) and str(row['Component']).strip():
        sections.append(f"[COMPONENT] {row['Component']}")

    # Add OS field
    if pd.notna(row['Op_sys']) and str(row['Op_sys']).strip():
        sections.append(f"[OS] {row['Op_sys']}")

    # Add PRIORITY field
    if pd.notna(row['Priority']) and str(row['Priority']).strip():
        sections.append(f"[PRIORITY] {row['Priority']}")

    # Add SUMMARY field (Title)
    if pd.notna(row['Title']) and str(row['Title']).strip():
        sections.append(f"[SUMMARY] {row['Title']}")

    # Add DESCRIPTION field
    # Use Enhanced_Description for VLM mode (includes VLM outputs)
    # Use Description for text-only mode
    if use_vlm:
        if pd.notna(row['Enhanced_Description']) and str(row['Enhanced_Description']).strip():
            sections.append(f"[DESCRIPTION] {row['Enhanced_Description']}")
    else:
        if pd.notna(row['Description']) and str(row['Description']).strip():
            sections.append(f"[DESCRIPTION] {row['Description']}")

    # Join all sections with newline
    return '\n'.join(sections)


def parse_duplicate_list(dup_str: str) -> Set[int]:
    """
    Parse semicolon-separated duplicate IDs.

    Args:
        dup_str: String like "123;456;789" or "NULL"

    Returns:
        Set of duplicate issue IDs
    """
    if pd.isna(dup_str) or str(dup_str).strip().upper() == 'NULL':
        return set()

    duplicates = set()
    for dup_id in str(dup_str).split(';'):
        dup_id = dup_id.strip()
        if dup_id and dup_id.upper() != 'NULL':
            try:
                duplicates.add(int(dup_id))
            except ValueError:
                print(f"Warning: Could not parse duplicate ID: {dup_id}")

    return duplicates


def build_duplicate_clusters(split_df: pd.DataFrame) -> Dict[int, int]:
    """
    Build duplicate clusters from split CSV and assign cluster IDs.

    Each cluster contains all bug reports that are duplicates of each other.
    We use Union-Find to merge duplicate groups.

    Args:
        split_df: DataFrame with Issue_id and Duplicate columns

    Returns:
        Dictionary mapping issue_id -> cluster_id
    """
    # Parse all duplicate relationships
    issue_to_duplicates = {}
    all_issues = set()

    for _, row in split_df.iterrows():
        issue_id = int(row['Issue_id'])
        all_issues.add(issue_id)
        duplicates = parse_duplicate_list(row['Duplicate'])
        issue_to_duplicates[issue_id] = duplicates

    # Union-Find data structure to merge duplicate groups
    parent = {issue_id: issue_id for issue_id in all_issues}

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])  # Path compression
        return parent[x]

    def union(x, y):
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y:
            parent[root_y] = root_x

    # Merge duplicate groups
    for issue_id, duplicates in issue_to_duplicates.items():
        for dup_id in duplicates:
            if dup_id in all_issues:
                union(issue_id, dup_id)

    # Assign cluster IDs (use the root as cluster ID)
    issue_to_cluster = {}
    for issue_id in all_issues:
        root = find(issue_id)
        issue_to_cluster[issue_id] = root

    return issue_to_cluster


def process_split(
    split_name: str,
    split_df: pd.DataFrame,
    bug_data: pd.DataFrame,
    output_dir: Path
) -> Dict[str, any]:
    """
    Process a single split (train/dev/test).

    Args:
        split_name: Name of the split ('train', 'dev', or 'test')
        split_df: DataFrame with Issue_id and Duplicate columns
        bug_data: DataFrame with all bug report data
        output_dir: Directory to save processed files

    Returns:
        Statistics dictionary
    """
    print(f"\nProcessing {split_name} split...")

    # Build duplicate clusters
    issue_to_cluster = build_duplicate_clusters(split_df)

    # Get issue IDs in this split
    split_issue_ids = set(split_df['Issue_id'].values)

    # Filter bug data to only include issues in this split
    split_bug_data = bug_data[bug_data['Issue_id'].isin(split_issue_ids)].copy()

    # Add cluster IDs
    split_bug_data['duplicate_cluster_id'] = split_bug_data['Issue_id'].map(issue_to_cluster)

    # Create augmented text (both VLM and text-only versions)
    print(f"Creating augmented text for {len(split_bug_data)} bug reports...")
    split_bug_data['augmented_text_with_vlm'] = split_bug_data.apply(
        lambda row: create_augmented_text(row, use_vlm=True), axis=1
    )
    split_bug_data['augmented_text_without_vlm'] = split_bug_data.apply(
        lambda row: create_augmented_text(row, use_vlm=False), axis=1
    )

    # Rename Issue_id to bug_id for consistency with existing code
    split_bug_data = split_bug_data.rename(columns={'Issue_id': 'bug_id'})

    # Select final columns
    final_columns = [
        'bug_id',
        'duplicate_cluster_id',
        'augmented_text_with_vlm',
        'augmented_text_without_vlm'
    ]
    final_data = split_bug_data[final_columns]

    # Save as CSV
    csv_path = output_dir / f"{split_name}.csv"
    final_data.to_csv(csv_path, index=False)
    print(f"Saved CSV to {csv_path}")

    # Save as JSON (for compatibility)
    json_path = output_dir / f"{split_name}.json"
    records = final_data.to_dict(orient='records')
    with open(json_path, 'w') as f:
        json.dump(records, f, indent=2)
    print(f"Saved JSON to {json_path}")

    # Calculate statistics
    cluster_counts = split_bug_data['duplicate_cluster_id'].value_counts()
    num_duplicates = (cluster_counts > 1).sum()
    num_singletons = (cluster_counts == 1).sum()
    total_duplicate_reports = cluster_counts[cluster_counts > 1].sum()

    stats = {
        'split': split_name,
        'total_reports': int(len(split_bug_data)),
        'duplicate_clusters': int(num_duplicates),
        'singleton_reports': int(num_singletons),
        'reports_in_duplicate_clusters': int(total_duplicate_reports),
        'avg_cluster_size': float(cluster_counts[cluster_counts > 1].mean()) if num_duplicates > 0 else 0.0,
        'max_cluster_size': int(cluster_counts.max())
    }

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess bug report data and create train/dev/test splits'
    )
    parser.add_argument(
        '--bug-data',
        type=str,
        default='data/processed_duplicate_training_data.csv',
        help='Path to processed bug report CSV file'
    )
    parser.add_argument(
        '--train-split',
        type=str,
        default='data/train/train.csv',
        help='Path to train split CSV file'
    )
    parser.add_argument(
        '--dev-split',
        type=str,
        default='data/dev/dev.csv',
        help='Path to dev split CSV file'
    )
    parser.add_argument(
        '--test-split',
        type=str,
        default='data/test/test.csv',
        help='Path to test split CSV file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed',
        help='Directory to save processed datasets'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("Bug Report Data Preprocessing")
    print("="*80)

    # Load main bug report data
    print(f"\nLoading bug report data from {args.bug_data}...")
    bug_data = pd.read_csv(args.bug_data)
    print(f"Loaded {len(bug_data)} bug reports")
    print(f"Columns: {list(bug_data.columns)}")

    # Load split files
    print(f"\nLoading split files...")
    train_split = pd.read_csv(args.train_split)
    dev_split = pd.read_csv(args.dev_split)
    test_split = pd.read_csv(args.test_split)

    print(f"Train split: {len(train_split)} issues")
    print(f"Dev split: {len(dev_split)} issues")
    print(f"Test split: {len(test_split)} issues")

    # Process each split
    all_stats = []

    for split_name, split_df in [
        ('train', train_split),
        ('dev', dev_split),
        ('test', test_split)
    ]:
        stats = process_split(split_name, split_df, bug_data, output_dir)
        all_stats.append(stats)

    # Print summary statistics
    print("\n" + "="*80)
    print("Summary Statistics")
    print("="*80)

    for stats in all_stats:
        print(f"\n{stats['split'].upper()} Split:")
        print(f"  Total reports: {stats['total_reports']}")
        print(f"  Duplicate clusters: {stats['duplicate_clusters']}")
        print(f"  Singleton reports (no duplicates): {stats['singleton_reports']}")
        print(f"  Reports in duplicate clusters: {stats['reports_in_duplicate_clusters']}")
        print(f"  Avg cluster size: {stats['avg_cluster_size']:.2f}")
        print(f"  Max cluster size: {stats['max_cluster_size']}")

    # Save statistics
    stats_path = output_dir / "preprocessing_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nSaved statistics to {stats_path}")

    print("\n" + "="*80)
    print("Preprocessing complete!")
    print("="*80)
    print(f"\nProcessed datasets saved to: {output_dir}/")
    print("Files created:")
    print(f"  - train.csv / train.json")
    print(f"  - dev.csv / dev.json")
    print(f"  - test.csv / test.json")
    print(f"  - preprocessing_stats.json")


if __name__ == '__main__':
    main()
