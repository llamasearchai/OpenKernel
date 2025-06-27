"""
Pytest configuration and fixtures for OpenKernel test suite.

This module provides shared fixtures and configuration for all tests,
including GPU detection, temporary directories, and mock objects.
"""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock
from typing import Generator, Dict, Any

# Import OpenKernel components for testing
from openkernel import (
    OpenKernelConfig,
    DistributedTrainer,
    InferenceEngine,
    DataPipeline,
    ResearchFramework,
    ModelArchitecture,
    TrainingMode
)


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU hardware"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "distributed: marks tests that require multiple processes"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark GPU tests based on function names."""
    for item in items:
        if "gpu" in item.name.lower() or "cuda" in item.name.lower():
            item.add_marker(pytest.mark.gpu)
        
        if "integration" in item.name.lower():
            item.add_marker(pytest.mark.integration)
        
        if "slow" in item.name.lower() or "benchmark" in item.name.lower():
            item.add_marker(pytest.mark.slow)


@pytest.fixture(scope="session")
def cuda_available() -> bool:
    """Check if CUDA is available for GPU tests."""
    try:
        import cupy
        return cupy.cuda.is_available()
    except ImportError:
        return False


@pytest.fixture(scope="session")
def skip_if_no_cuda(cuda_available):
    """Skip test if CUDA is not available."""
    if not cuda_available:
        pytest.skip("CUDA not available")


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def config() -> OpenKernelConfig:
    """Provide a default test configuration."""
    return OpenKernelConfig(
        model_size="7B",
        sequence_length=2048,  # Smaller for faster tests
        num_nodes=1,
        gpus_per_node=1,
        batch_size=2,
        training_mode=TrainingMode.PRETRAINING,
        model_architecture=ModelArchitecture.TRANSFORMER,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1
    )


@pytest.fixture
def large_config() -> OpenKernelConfig:
    """Provide a configuration for large-scale tests."""
    return OpenKernelConfig(
        model_size="70B",
        sequence_length=32768,
        num_nodes=8,
        gpus_per_node=8,
        batch_size=1024,
        training_mode=TrainingMode.PRETRAINING,
        model_architecture=ModelArchitecture.TRANSFORMER
    )


@pytest.fixture
def mock_gpu_environment(monkeypatch):
    """Mock GPU environment variables for testing."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "12345")


@pytest.fixture
def mock_distributed_environment(monkeypatch):
    """Mock distributed training environment."""
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("MASTER_ADDR", "10.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29500")


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = Mock()
    model.parameters.return_value = [Mock() for _ in range(10)]
    model.forward.return_value = Mock()
    model.state_dict.return_value = {"layer.weight": Mock()}
    return model


@pytest.fixture
def mock_optimizer():
    """Create a mock optimizer for testing."""
    optimizer = Mock()
    optimizer.step.return_value = None
    optimizer.zero_grad.return_value = None
    optimizer.state_dict.return_value = {"param_groups": []}
    return optimizer


@pytest.fixture
def mock_cuda_kernel():
    """Create a mock CUDA kernel for testing."""
    kernel = Mock()
    kernel.code = """
    extern "C" __global__ void test_kernel(float* a, float* b, float* c, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            c[idx] = a[idx] + b[idx];
        }
    }
    """
    kernel.compile.return_value = True
    kernel.execute.return_value = Mock()
    kernel.benchmark.return_value = {"execution_time": 1.5, "memory_usage": 1024}
    return kernel


@pytest.fixture
def sample_training_data():
    """Generate sample training data for tests."""
    import numpy as np
    
    batch_size = 4
    sequence_length = 128
    vocab_size = 1000
    
    input_ids = np.random.randint(0, vocab_size, (batch_size, sequence_length))
    attention_mask = np.ones((batch_size, sequence_length))
    labels = np.random.randint(0, vocab_size, (batch_size, sequence_length))
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


@pytest.fixture
def sample_inference_data():
    """Generate sample inference data for tests."""
    return [
        "Explain the fundamentals of quantum computing",
        "What are the key principles of machine learning?",
        "How do transformer models work?",
        "Describe CUDA kernel optimization techniques"
    ]


@pytest.fixture
def trainer(config, mock_gpu_environment):
    """Create a DistributedTrainer instance for testing."""
    return DistributedTrainer(config)


@pytest.fixture
def inference_engine(config):
    """Create an InferenceEngine instance for testing."""
    return InferenceEngine(config)


@pytest.fixture
def data_pipeline(config, temp_dir):
    """Create a DataPipeline instance for testing."""
    config.data_dir = str(temp_dir)
    return DataPipeline(config)


@pytest.fixture
def research_framework(config):
    """Create a ResearchFramework instance for testing."""
    return ResearchFramework(config)


@pytest.fixture
def mock_web_content():
    """Mock web content for data pipeline testing."""
    return {
        "url": "https://example.com/article",
        "title": "Sample Article Title",
        "content": "This is sample content for testing the data pipeline. " * 100,
        "metadata": {
            "author": "Test Author",
            "date": "2024-01-15",
            "language": "en"
        }
    }


@pytest.fixture
def performance_metrics():
    """Sample performance metrics for testing."""
    return {
        "training": {
            "loss": 2.5,
            "learning_rate": 1e-4,
            "grad_norm": 1.2,
            "tokens_per_second": 1500,
            "gpu_utilization": 0.85,
            "memory_usage": 0.75
        },
        "inference": {
            "latency_p50": 35.2,
            "latency_p95": 89.5,
            "latency_p99": 145.8,
            "throughput": 2500,
            "tokens_per_second": 1800,
            "memory_efficiency": 0.68
        }
    }


@pytest.fixture(autouse=True)
def setup_test_environment(temp_dir, monkeypatch):
    """Set up test environment with temporary directories."""
    # Create test directories
    test_dirs = ["logs", "checkpoints", "cache", "results", "data"]
    for dir_name in test_dirs:
        (temp_dir / dir_name).mkdir(exist_ok=True)
    
    # Set environment variables for testing
    monkeypatch.setenv("OPENKERNEL_TEST_MODE", "1")
    monkeypatch.setenv("OPENKERNEL_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("OPENKERNEL_CACHE_DIR", str(temp_dir / "cache"))
    monkeypatch.setenv("OPENKERNEL_LOG_DIR", str(temp_dir / "logs"))


@pytest.fixture
def benchmark_config():
    """Configuration for benchmark tests."""
    return {
        "warmup_iterations": 3,
        "measurement_iterations": 10,
        "timeout": 60.0,
        "memory_limit": "8GB",
        "gpu_memory_limit": "16GB"
    }


class MockLogger:
    """Mock logger for testing."""
    
    def __init__(self):
        self.messages = []
    
    def info(self, msg, *args, **kwargs):
        self.messages.append(("INFO", msg))
    
    def debug(self, msg, *args, **kwargs):
        self.messages.append(("DEBUG", msg))
    
    def warning(self, msg, *args, **kwargs):
        self.messages.append(("WARNING", msg))
    
    def error(self, msg, *args, **kwargs):
        self.messages.append(("ERROR", msg))


@pytest.fixture
def mock_logger():
    """Provide a mock logger for testing."""
    return MockLogger()


@pytest.fixture
def gpu_memory_info():
    """Mock GPU memory information."""
    return {
        "total": 16 * 1024**3,  # 16GB
        "available": 12 * 1024**3,  # 12GB available
        "used": 4 * 1024**3,  # 4GB used
        "utilization": 0.25
    }


@pytest.fixture(scope="session")
def test_data_samples():
    """Generate various test data samples."""
    import numpy as np
    
    return {
        "small_matrix": np.random.randn(64, 64).astype(np.float32),
        "large_matrix": np.random.randn(1024, 1024).astype(np.float32),
        "text_samples": [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming technology.",
            "CUDA kernels enable high-performance computing.",
            "Distributed training scales to multiple GPUs."
        ],
        "numeric_data": np.random.randn(1000, 128).astype(np.float32),
        "labels": np.random.randint(0, 10, 1000)
    }


# Pytest plugins and hooks
def pytest_runtest_setup(item):
    """Setup before each test."""
    # Skip GPU tests if CUDA not available
    if item.get_closest_marker("gpu"):
        try:
            import cupy
            if not cupy.cuda.is_available():
                pytest.skip("CUDA not available")
        except ImportError:
            pytest.skip("CuPy not installed")
    
    # Skip slow tests in fast mode
    if item.get_closest_marker("slow") and item.config.getoption("-m") == "not slow":
        pytest.skip("Slow test skipped in fast mode")


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-slow", action="store_true", default=False,
        help="Run slow tests"
    )
    parser.addoption(
        "--run-gpu", action="store_true", default=False,
        help="Run GPU tests"
    )
    parser.addoption(
        "--benchmark", action="store_true", default=False,
        help="Run benchmark tests"
    )


 