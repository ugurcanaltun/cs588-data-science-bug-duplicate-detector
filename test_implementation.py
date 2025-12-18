"""
Comprehensive test script for the duplicate bug detection implementation.

This script tests all components to ensure they work correctly:
- Model architecture
- Loss functions
- Evaluation metrics
- Data loading
- Training integration
- Evaluation integration
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

import torch
import numpy as np
import pandas as pd

# Color codes for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_test(msg):
    print(f"{Colors.BLUE}[TEST]{Colors.END} {msg}")

def print_success(msg):
    print(f"{Colors.GREEN}[PASS]{Colors.END} {msg}")

def print_error(msg):
    print(f"{Colors.RED}[FAIL]{Colors.END} {msg}")

def print_warning(msg):
    print(f"{Colors.YELLOW}[WARN]{Colors.END} {msg}")


class TestRunner:
    """Test runner for all components."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.temp_dir = None

    def setup(self):
        """Setup test environment."""
        print_test("Setting up test environment...")
        self.temp_dir = tempfile.mkdtemp()
        print_success(f"Created temporary directory: {self.temp_dir}")

    def teardown(self):
        """Clean up test environment."""
        print_test("Cleaning up test environment...")
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print_success("Cleaned up temporary directory")

    def run_test(self, test_func, test_name):
        """Run a single test."""
        print_test(f"Running: {test_name}")
        try:
            test_func()
            self.passed += 1
            print_success(f"PASSED: {test_name}\n")
        except Exception as e:
            self.failed += 1
            print_error(f"FAILED: {test_name}")
            print_error(f"Error: {str(e)}\n")
            import traceback
            traceback.print_exc()

    def print_summary(self):
        """Print test summary."""
        total = self.passed + self.failed
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"Total tests: {total}")
        print(f"{Colors.GREEN}Passed: {self.passed}{Colors.END}")
        print(f"{Colors.RED}Failed: {self.failed}{Colors.END}")
        print(f"Success rate: {100*self.passed/total:.1f}%")
        print("="*60)

        if self.failed == 0:
            print(f"\n{Colors.GREEN}All tests passed!{Colors.END}\n")
        else:
            print(f"\n{Colors.RED}Some tests failed. Please review errors above.{Colors.END}\n")


# ============================================================================
# TEST UTILITIES
# ============================================================================

def create_mock_data(num_reports=100, num_clusters=20):
    """Create mock bug report data for testing."""
    data = []
    cluster_ids = np.random.randint(0, num_clusters, size=num_reports)

    for i in range(num_reports):
        bug_id = 10000 + i
        cluster_id = int(cluster_ids[i])

        # Create mock augmented text
        text_with_vlm = f"""[PRODUCT] TestProduct
[COMPONENT] TestComponent
[VERSION] {i % 10}
[OS] TestOS
[SUMMARY] Test bug report {i} in cluster {cluster_id}
[DESCRIPTION] This is a test description for bug {i}
[SCREENSHOT_CAPTION] Mock screenshot caption for bug {i}
[SCREENSHOT_ERROR_TEXT] Mock error text for bug {i}"""

        text_without_vlm = f"""[PRODUCT] TestProduct
[COMPONENT] TestComponent
[VERSION] {i % 10}
[OS] TestOS
[SUMMARY] Test bug report {i} in cluster {cluster_id}
[DESCRIPTION] This is a test description for bug {i}"""

        data.append({
            'bug_id': bug_id,
            'duplicate_cluster_id': cluster_id,
            'augmented_text_with_vlm': text_with_vlm,
            'augmented_text_without_vlm': text_without_vlm
        })

    return data


# ============================================================================
# MODEL TESTS
# ============================================================================

def test_model_initialization():
    """Test model initialization."""
    from src.model import BugReportEncoder

    # Test default initialization
    model = BugReportEncoder()
    assert model is not None, "Model should be initialized"
    assert model.embedding_dim > 0, "Embedding dimension should be positive"
    print(f"  Model initialized with embedding_dim={model.embedding_dim}")

    # Test frozen mode
    model_frozen = BugReportEncoder(freeze=True)
    for param in model_frozen.encoder.parameters():
        assert not param.requires_grad, "Frozen model parameters should not require gradients"
    print("  Frozen mode works correctly")


def test_model_forward():
    """Test model forward pass."""
    from src.model import BugReportEncoder

    model = BugReportEncoder()
    texts = [
        "[SUMMARY] Test bug 1",
        "[SUMMARY] Test bug 2",
        "[SUMMARY] Test bug 3"
    ]

    embeddings = model(texts)

    assert embeddings.shape[0] == len(texts), "Batch size mismatch"
    assert embeddings.shape[1] == model.embedding_dim, "Embedding dimension mismatch"
    assert torch.isfinite(embeddings).all(), "Embeddings should be finite"

    # Check normalization
    norms = torch.norm(embeddings, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
        "Embeddings should be L2-normalized"

    print(f"  Forward pass produces embeddings of shape {embeddings.shape}")


def test_model_save_load(temp_dir):
    """Test model save and load."""
    from src.model import BugReportEncoder

    # Create and save model
    model = BugReportEncoder()
    save_path = os.path.join(temp_dir, "test_model")
    model.save_pretrained(save_path)

    assert os.path.exists(save_path), "Model directory should exist"
    print(f"  Model saved to {save_path}")

    # Load model
    loaded_model = BugReportEncoder.load_pretrained(save_path)
    assert loaded_model.embedding_dim == model.embedding_dim, \
        "Loaded model should have same embedding dimension"

    # Test that loaded model produces same outputs
    texts = ["[SUMMARY] Test bug"]
    orig_emb = model(texts)
    loaded_emb = loaded_model(texts)

    assert torch.allclose(orig_emb, loaded_emb, atol=1e-5), \
        "Loaded model should produce same embeddings"

    print("  Model save/load works correctly")


# ============================================================================
# LOSS TESTS
# ============================================================================

def test_contrastive_loss():
    """Test supervised contrastive loss."""
    from src.loss import SupervisedContrastiveLoss

    criterion = SupervisedContrastiveLoss(temperature=0.07)

    # Create mock embeddings and labels
    batch_size = 16
    embedding_dim = 128
    embeddings = torch.randn(batch_size, embedding_dim)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    # Create labels with duplicates
    labels = torch.tensor([0, 0, 1, 1, 1, 2, 2, 3, 4, 4, 5, 6, 6, 6, 7, 8])

    loss, metrics = criterion(embeddings, labels)

    assert loss.item() >= 0, "Loss should be non-negative"
    assert torch.isfinite(loss), "Loss should be finite"
    assert metrics['num_valid_anchors'] > 0, "Should have valid anchors"
    assert metrics['avg_num_positives'] > 0, "Should have positives"

    print(f"  Loss: {loss.item():.4f}")
    print(f"  Valid anchors: {metrics['num_valid_anchors']}")
    print(f"  Avg positives: {metrics['avg_num_positives']:.2f}")


def test_contrastive_loss_edge_cases():
    """Test contrastive loss edge cases."""
    from src.loss import SupervisedContrastiveLoss

    criterion = SupervisedContrastiveLoss()

    # Case 1: No duplicates (all different labels)
    embeddings = torch.randn(8, 128)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    labels = torch.arange(8)

    loss, metrics = criterion(embeddings, labels)
    assert loss.item() == 0.0, "Loss should be 0 when no duplicates"
    assert metrics['num_valid_anchors'] == 0, "Should have no valid anchors"
    print("  Edge case (no duplicates) handled correctly")

    # Case 2: All same label
    labels = torch.zeros(8, dtype=torch.long)
    loss, metrics = criterion(embeddings, labels)
    assert loss.item() > 0, "Loss should be positive when all are duplicates"
    print("  Edge case (all duplicates) handled correctly")


def test_triplet_loss():
    """Test triplet loss."""
    from src.loss import TripletLoss

    criterion = TripletLoss(margin=0.5)

    embeddings = torch.randn(16, 128)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    labels = torch.tensor([0, 0, 1, 1, 1, 2, 2, 3, 4, 4, 5, 6, 6, 6, 7, 8])

    loss, metrics = criterion(embeddings, labels)

    assert loss.item() >= 0, "Loss should be non-negative"
    assert torch.isfinite(loss), "Loss should be finite"

    print(f"  Triplet loss: {loss.item():.4f}")


# ============================================================================
# METRICS TESTS
# ============================================================================

def test_recall_at_k():
    """Test Recall@k metric."""
    from src.metrics import RetrievalMetrics

    metrics_calc = RetrievalMetrics(k_values=[1, 5, 10])

    # Mock ranked candidates and relevant IDs
    ranked_candidates = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    relevant_ids = {30, 70}

    recall_1 = metrics_calc.compute_recall_at_k(ranked_candidates, relevant_ids, 1)
    assert recall_1 == 0.0, "Recall@1 should be 0 (no relevant in top-1)"

    recall_5 = metrics_calc.compute_recall_at_k(ranked_candidates, relevant_ids, 5)
    assert recall_5 == 1.0, "Recall@5 should be 1 (relevant at position 3)"

    recall_10 = metrics_calc.compute_recall_at_k(ranked_candidates, relevant_ids, 10)
    assert recall_10 == 1.0, "Recall@10 should be 1 (both relevant in top-10)"

    print("  Recall@k computed correctly")


def test_mrr():
    """Test Mean Reciprocal Rank."""
    from src.metrics import RetrievalMetrics

    metrics_calc = RetrievalMetrics()

    # Case 1: Relevant at position 3
    ranked_candidates = [10, 20, 30, 40, 50]
    relevant_ids = {30}
    mrr = metrics_calc.compute_reciprocal_rank(ranked_candidates, relevant_ids)
    assert abs(mrr - 1/3) < 1e-6, f"MRR should be 1/3, got {mrr}"

    # Case 2: Relevant at position 1
    relevant_ids = {10}
    mrr = metrics_calc.compute_reciprocal_rank(ranked_candidates, relevant_ids)
    assert mrr == 1.0, "MRR should be 1.0"

    # Case 3: No relevant
    relevant_ids = {99}
    mrr = metrics_calc.compute_reciprocal_rank(ranked_candidates, relevant_ids)
    assert mrr == 0.0, "MRR should be 0.0"

    print("  MRR computed correctly")


def test_map_at_k():
    """Test Mean Average Precision at k."""
    from src.metrics import RetrievalMetrics

    metrics_calc = RetrievalMetrics()

    # Relevant at positions 2, 4, 7
    ranked_candidates = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    relevant_ids = {20, 40, 70}

    map_10 = metrics_calc.compute_average_precision_at_k(
        ranked_candidates, relevant_ids, 10
    )

    # Manual calculation: (1/2 + 2/4 + 3/7) / 3
    expected = (0.5 + 0.5 + 3/7) / 3
    assert abs(map_10 - expected) < 1e-6, f"MAP@10 should be {expected}, got {map_10}"

    print(f"  MAP@k computed correctly: {map_10:.4f}")


def test_compute_metrics():
    """Test full metrics computation."""
    from src.metrics import RetrievalMetrics, compute_similarity_matrix

    # Create mock embeddings and labels
    num_reports = 50
    num_clusters = 10
    embedding_dim = 128

    embeddings = torch.randn(num_reports, embedding_dim)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    cluster_ids = np.random.randint(0, num_clusters, size=num_reports).tolist()
    bug_ids = list(range(10000, 10000 + num_reports))

    # Compute similarity matrix
    similarities = compute_similarity_matrix(embeddings, embeddings)

    assert similarities.shape == (num_reports, num_reports), "Similarity matrix shape mismatch"
    assert torch.allclose(similarities.diagonal(), torch.ones(num_reports), atol=1e-5), \
        "Diagonal should be 1 (self-similarity)"

    # Compute metrics
    metrics_calc = RetrievalMetrics(k_values=[1, 5, 10])
    metrics = metrics_calc.compute_metrics(similarities, cluster_ids, bug_ids)

    # Check that all metrics are present
    for k in [1, 5, 10]:
        assert f'recall@{k}' in metrics, f"recall@{k} should be in metrics"
        assert f'map@{k}' in metrics, f"map@{k} should be in metrics"
    assert 'mrr' in metrics, "MRR should be in metrics"
    assert 'num_queries' in metrics, "num_queries should be in metrics"

    # Check that metrics are in valid range
    for key, value in metrics.items():
        if key != 'num_queries':
            assert 0 <= value <= 1, f"{key} should be in [0, 1], got {value}"

    print(f"  Full metrics computed successfully")
    print(f"  Recall@10: {metrics['recall@10']:.4f}")
    print(f"  MRR: {metrics['mrr']:.4f}")
    print(f"  MAP@10: {metrics['map@10']:.4f}")


# ============================================================================
# DATA TESTS
# ============================================================================

def test_dataset_creation(temp_dir):
    """Test dataset creation and loading."""
    from src.data import BugReportDataset

    # Create mock data
    data = create_mock_data(num_reports=100, num_clusters=20)
    data_path = os.path.join(temp_dir, "test_data.json")

    with open(data_path, 'w') as f:
        json.dump(data, f)

    # Test loading with VLM
    dataset_vlm = BugReportDataset(data_path, use_vlm_augmentation=True)
    assert len(dataset_vlm) == 100, "Dataset should have 100 reports"

    # Test loading without VLM
    dataset_no_vlm = BugReportDataset(data_path, use_vlm_augmentation=False)
    assert len(dataset_no_vlm) == 100, "Dataset should have 100 reports"

    # Test getitem
    item = dataset_vlm[0]
    assert 'bug_id' in item, "Item should have bug_id"
    assert 'text' in item, "Item should have text"
    assert 'cluster_id' in item, "Item should have cluster_id"

    # Check that VLM text is longer
    assert len(dataset_vlm[0]['text']) > len(dataset_no_vlm[0]['text']), \
        "VLM-augmented text should be longer"

    print(f"  Dataset loaded: {len(dataset_vlm)} reports")
    print(f"  Cluster mapping: {len(dataset_vlm.cluster_to_indices)} clusters")


def test_cluster_sampler(temp_dir):
    """Test cluster-balanced batch sampler."""
    from src.data import BugReportDataset, ClusterBalancedBatchSampler

    # Create mock data with clear cluster structure
    data = create_mock_data(num_reports=100, num_clusters=10)
    data_path = os.path.join(temp_dir, "test_data.json")

    with open(data_path, 'w') as f:
        json.dump(data, f)

    dataset = BugReportDataset(data_path, use_vlm_augmentation=True)

    # Create sampler
    sampler = ClusterBalancedBatchSampler(
        dataset=dataset,
        batch_size=16,
        samples_per_cluster=4,
        drop_last=True
    )

    # Test batch generation
    batches = list(sampler)
    assert len(batches) > 0, "Should generate batches"

    # Check batch properties
    for batch_indices in batches[:3]:  # Check first 3 batches
        assert len(batch_indices) <= 16, "Batch size should not exceed 16"

        # Get cluster IDs for this batch
        batch_clusters = [dataset.data.iloc[idx]['duplicate_cluster_id']
                         for idx in batch_indices]

        # Count occurrences of each cluster
        from collections import Counter
        cluster_counts = Counter(batch_clusters)

        # Each cluster should appear multiple times (for contrastive learning)
        for cluster_id, count in cluster_counts.items():
            assert count >= 2, f"Cluster {cluster_id} should appear at least twice in batch"

    print(f"  Cluster-balanced sampler generated {len(batches)} batches")
    print(f"  Each batch has multiple samples from same clusters")


def test_dataloader(temp_dir):
    """Test dataloader creation."""
    from src.data import create_dataloader

    # Create mock data
    data = create_mock_data(num_reports=100, num_clusters=20)
    data_path = os.path.join(temp_dir, "test_data.json")

    with open(data_path, 'w') as f:
        json.dump(data, f)

    # Create dataloader with cluster sampling
    dataloader = create_dataloader(
        data_path=data_path,
        batch_size=16,
        use_vlm_augmentation=True,
        use_cluster_sampling=True,
        samples_per_cluster=4,
        num_workers=0
    )

    # Test batch
    texts, cluster_ids, bug_ids = next(iter(dataloader))

    assert isinstance(texts, list), "Texts should be a list"
    assert isinstance(cluster_ids, torch.Tensor), "Cluster IDs should be a tensor"
    assert isinstance(bug_ids, list), "Bug IDs should be a list"
    assert len(texts) == len(cluster_ids) == len(bug_ids), "Batch sizes should match"

    print(f"  Dataloader created successfully")
    print(f"  First batch: {len(texts)} samples")


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_training_iteration(temp_dir):
    """Test a single training iteration."""
    from src.model import BugReportEncoder
    from src.loss import SupervisedContrastiveLoss
    from src.data import create_dataloader

    # Create mock data
    data = create_mock_data(num_reports=100, num_clusters=20)
    data_path = os.path.join(temp_dir, "train_data.json")

    with open(data_path, 'w') as f:
        json.dump(data, f)

    # Create model, loss, optimizer
    model = BugReportEncoder()
    criterion = SupervisedContrastiveLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # Create dataloader
    dataloader = create_dataloader(
        data_path=data_path,
        batch_size=16,
        use_vlm_augmentation=True,
        use_cluster_sampling=True,
        num_workers=0
    )

    # Training iteration
    model.train()
    texts, cluster_ids, _ = next(iter(dataloader))

    embeddings = model(texts)
    loss, metrics = criterion(embeddings, cluster_ids)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert loss.item() > 0, "Loss should be positive"
    assert torch.isfinite(loss), "Loss should be finite"

    print(f"  Training iteration successful")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Valid anchors: {metrics['num_valid_anchors']}")


def test_evaluation_pipeline(temp_dir):
    """Test evaluation pipeline."""
    from src.model import BugReportEncoder
    from src.data import load_data_for_evaluation
    from src.metrics import RetrievalMetrics, compute_similarity_matrix

    # Create mock data
    data = create_mock_data(num_reports=50, num_clusters=10)
    data_path = os.path.join(temp_dir, "eval_data.json")

    with open(data_path, 'w') as f:
        json.dump(data, f)

    # Load data
    texts, cluster_ids, bug_ids = load_data_for_evaluation(
        data_path=data_path,
        use_vlm_augmentation=True
    )

    assert len(texts) == 50, "Should load 50 texts"
    assert len(cluster_ids) == 50, "Should load 50 cluster IDs"
    assert len(bug_ids) == 50, "Should load 50 bug IDs"

    # Encode
    model = BugReportEncoder()
    model.eval()

    with torch.no_grad():
        embeddings = model.encode_batch(texts, batch_size=16)

    assert embeddings.shape[0] == 50, "Should have 50 embeddings"

    # Compute similarities
    similarities = compute_similarity_matrix(embeddings, embeddings)

    # Compute metrics
    metrics_calc = RetrievalMetrics(k_values=[1, 5, 10])
    metrics = metrics_calc.compute_metrics(similarities, cluster_ids, bug_ids)

    assert 'recall@10' in metrics, "Should compute Recall@10"
    assert 'mrr' in metrics, "Should compute MRR"
    assert 'map@10' in metrics, "Should compute MAP@10"

    print(f"  Evaluation pipeline successful")
    print(f"  Metrics: Recall@10={metrics['recall@10']:.4f}, MRR={metrics['mrr']:.4f}")


def test_model_save_load_integration(temp_dir):
    """Test save/load in full pipeline."""
    from src.model import BugReportEncoder

    # Create and train a bit
    model = BugReportEncoder()
    texts = ["[SUMMARY] Test 1", "[SUMMARY] Test 2"]

    # Get embeddings before saving
    model.eval()
    with torch.no_grad():
        embeddings_before = model(texts)

    # Save
    save_path = os.path.join(temp_dir, "integration_model")
    model.save_pretrained(save_path)

    # Load
    loaded_model = BugReportEncoder.load_pretrained(save_path)
    loaded_model.eval()

    # Get embeddings after loading
    with torch.no_grad():
        embeddings_after = loaded_model(texts)

    # Should be identical
    assert torch.allclose(embeddings_before, embeddings_after, atol=1e-5), \
        "Embeddings should match after save/load"

    print(f"  Save/load integration successful")


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("DUPLICATE BUG DETECTION - IMPLEMENTATION TESTS")
    print("="*60 + "\n")

    runner = TestRunner()
    runner.setup()

    try:
        # Model tests
        print(f"\n{Colors.BLUE}=== MODEL TESTS ==={Colors.END}\n")
        runner.run_test(test_model_initialization, "Model Initialization")
        runner.run_test(test_model_forward, "Model Forward Pass")
        runner.run_test(lambda: test_model_save_load(runner.temp_dir), "Model Save/Load")

        # Loss tests
        print(f"\n{Colors.BLUE}=== LOSS FUNCTION TESTS ==={Colors.END}\n")
        runner.run_test(test_contrastive_loss, "Supervised Contrastive Loss")
        runner.run_test(test_contrastive_loss_edge_cases, "Contrastive Loss Edge Cases")
        runner.run_test(test_triplet_loss, "Triplet Loss")

        # Metrics tests
        print(f"\n{Colors.BLUE}=== METRICS TESTS ==={Colors.END}\n")
        runner.run_test(test_recall_at_k, "Recall@k")
        runner.run_test(test_mrr, "Mean Reciprocal Rank")
        runner.run_test(test_map_at_k, "Mean Average Precision@k")
        runner.run_test(test_compute_metrics, "Full Metrics Computation")

        # Data tests
        print(f"\n{Colors.BLUE}=== DATA LOADING TESTS ==={Colors.END}\n")
        runner.run_test(lambda: test_dataset_creation(runner.temp_dir), "Dataset Creation")
        runner.run_test(lambda: test_cluster_sampler(runner.temp_dir), "Cluster-Balanced Sampler")
        runner.run_test(lambda: test_dataloader(runner.temp_dir), "DataLoader Creation")

        # Integration tests
        print(f"\n{Colors.BLUE}=== INTEGRATION TESTS ==={Colors.END}\n")
        runner.run_test(lambda: test_training_iteration(runner.temp_dir), "Training Iteration")
        runner.run_test(lambda: test_evaluation_pipeline(runner.temp_dir), "Evaluation Pipeline")
        runner.run_test(lambda: test_model_save_load_integration(runner.temp_dir),
                       "Save/Load Integration")

    finally:
        runner.teardown()

    runner.print_summary()

    # Return exit code
    return 0 if runner.failed == 0 else 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
