"""
Script to compare results across different experiments.

This script loads evaluation metrics from multiple experiments and
generates a comparison table showing Recall@k, MRR, and MAP@k.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List
import pandas as pd


def load_metrics(metrics_path: str) -> Dict:
    """Load metrics from JSON file."""
    with open(metrics_path, 'r') as f:
        return json.load(f)


def find_metrics_files(output_dir: str, filename: str = "metrics_with_vlm.json") -> Dict[str, str]:
    """
    Find all metrics files in the output directory.

    Returns:
        Dictionary mapping experiment names to metrics file paths
    """
    output_path = Path(output_dir)
    metrics_files = {}

    # Search for metrics files
    for path in output_path.rglob(filename):
        # Extract experiment name from path
        parts = path.parts
        if output_dir in parts:
            idx = parts.index(Path(output_dir).name)
            if idx + 1 < len(parts):
                exp_name = parts[idx + 1]
                metrics_files[exp_name] = str(path)

    return metrics_files


def create_comparison_table(experiments: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create a comparison table from multiple experiments.

    Args:
        experiments: Dictionary mapping experiment names to metrics

    Returns:
        DataFrame with comparison results
    """
    rows = []

    for exp_name, metrics in experiments.items():
        row = {'Experiment': exp_name}

        # Add Recall@k
        for k in [1, 5, 10, 20]:
            key = f'recall@{k}'
            if key in metrics:
                row[f'Recall@{k}'] = f"{metrics[key]:.4f}"

        # Add MRR
        if 'mrr' in metrics:
            row['MRR'] = f"{metrics['mrr']:.4f}"

        # Add MAP@k
        for k in [1, 5, 10, 20]:
            key = f'map@{k}'
            if key in metrics:
                row[f'MAP@{k}'] = f"{metrics[key]:.4f}"

        # Add number of queries
        if 'num_queries' in metrics:
            row['Num Queries'] = int(metrics['num_queries'])

        rows.append(row)

    df = pd.DataFrame(rows)

    # Reorder columns
    cols = ['Experiment']
    recall_cols = [c for c in df.columns if c.startswith('Recall@')]
    cols.extend(sorted(recall_cols, key=lambda x: int(x.split('@')[1])))
    cols.append('MRR')
    map_cols = [c for c in df.columns if c.startswith('MAP@')]
    cols.extend(sorted(map_cols, key=lambda x: int(x.split('@')[1])))
    if 'Num Queries' in df.columns:
        cols.append('Num Queries')

    return df[cols]


def print_latex_table(df: pd.DataFrame):
    """Print DataFrame in LaTeX table format."""
    print("\nLaTeX Table Format:")
    print("=" * 80)
    print(df.to_latex(index=False, escape=False))
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Compare results across experiments")

    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory containing experiment results')
    parser.add_argument('--experiments', type=str, nargs='+', default=None,
                        help='List of experiment names to compare (default: all found)')
    parser.add_argument('--use_vlm', action='store_true',
                        help='Compare VLM-augmented results (default: without VLM)')
    parser.add_argument('--save_csv', type=str, default=None,
                        help='Save comparison table to CSV file')
    parser.add_argument('--latex', action='store_true',
                        help='Print LaTeX table format')

    args = parser.parse_args()

    # Determine which metrics file to look for
    metrics_filename = "metrics_with_vlm.json" if args.use_vlm else "metrics_without_vlm.json"

    print(f"Searching for experiments in: {args.output_dir}")
    print(f"Looking for: {metrics_filename}")
    print()

    # Find all metrics files
    if args.experiments is None:
        # Auto-discover experiments
        all_metrics = find_metrics_files(args.output_dir, metrics_filename)
        if not all_metrics:
            print(f"No metrics files found in {args.output_dir}")
            print("Make sure you have run experiments and they contain evaluation results.")
            return

        print(f"Found {len(all_metrics)} experiments:")
        for exp_name in all_metrics.keys():
            print(f"  - {exp_name}")
        print()
    else:
        # Use specified experiments
        all_metrics = {}
        for exp_name in args.experiments:
            metrics_path = Path(args.output_dir) / exp_name / 'evaluation_results' / metrics_filename
            if metrics_path.exists():
                all_metrics[exp_name] = str(metrics_path)
            else:
                print(f"Warning: Metrics not found for {exp_name} at {metrics_path}")

    if not all_metrics:
        print("No valid experiments found.")
        return

    # Load metrics for all experiments
    experiments = {}
    for exp_name, metrics_path in all_metrics.items():
        try:
            metrics = load_metrics(metrics_path)
            experiments[exp_name] = metrics
        except Exception as e:
            print(f"Warning: Could not load metrics for {exp_name}: {e}")

    if not experiments:
        print("No metrics could be loaded.")
        return

    # Create comparison table
    print("Creating comparison table...")
    df = create_comparison_table(experiments)

    # Print table
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPARISON")
    print("=" * 80)
    print()
    print(df.to_string(index=False))
    print()
    print("=" * 80)

    # Print LaTeX format if requested
    if args.latex:
        print_latex_table(df)

    # Save to CSV if requested
    if args.save_csv:
        df.to_csv(args.save_csv, index=False)
        print(f"\nSaved comparison table to: {args.save_csv}")

    # Print summary statistics
    print("\nSummary:")
    print(f"  Total experiments compared: {len(experiments)}")
    print(f"  VLM augmentation: {'Yes' if args.use_vlm else 'No'}")

    # Find best performing experiment for key metrics
    numeric_df = df.copy()
    for col in numeric_df.columns:
        if col != 'Experiment' and col != 'Num Queries':
            numeric_df[col] = numeric_df[col].astype(float)

    print("\nBest performing experiments:")
    for metric in ['Recall@10', 'MRR', 'MAP@10']:
        if metric in numeric_df.columns:
            best_idx = numeric_df[metric].idxmax()
            best_exp = numeric_df.loc[best_idx, 'Experiment']
            best_val = numeric_df.loc[best_idx, metric]
            print(f"  {metric}: {best_exp} ({best_val:.4f})")


if __name__ == '__main__':
    main()
