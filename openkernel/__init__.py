#!/usr/bin/env python3
"""
OpenKernel - Advanced CUDA Kernel Development & AI Training Infrastructure

A comprehensive toolkit for AI infrastructure engineering, featuring:
- Advanced CUDA kernel development and optimization
- Distributed training for trillion-parameter models
- Inference optimization with novel architectures
- Internet-scale data pipeline processing
- Research frameworks for architecture comparison
- Production deployment with monitoring and scaling
"""

__version__ = "1.0.0"
__author__ = "OpenKernel Team"

# Core imports
from .core import (
    OpenKernelConfig,
    ModelArchitecture,
    TrainingMode,
    DatasetType,
    OptimizationLevel,
    PrecisionType,
    TrainingMetrics,
    InferenceMetrics,
    SystemMetrics
)

# Try to import optional modules gracefully
try:
    from .training import (
        DistributedTrainer,
        ModelParallelTrainer,
        OptimizerFactory,
        LearningRateScheduler
    )
except ImportError:
    pass

try:
    from .inference import (
        InferenceEngine,
        KVCache,
        SpeculativeDecoder,
        BatchProcessor
    )
except ImportError:
    pass

try:
    from .data import (
        DataPipeline,
        WebCrawler,
        DataProcessor,
        QualityFilter
    )
except ImportError:
    pass

try:
    from .cuda import (
        CUDAKernelGenerator,
        KernelOptimizer,
        PerformanceBenchmark
    )
except ImportError:
    pass

try:
    from .research import (
        ResearchFramework,
        ExperimentManager,
        ArchitectureComparator,
        LongContextEvaluator
    )
except ImportError:
    pass

try:
    from .monitoring import (
        MetricsCollector,
        AlertManager,
        PerformanceProfiler
    )
except ImportError:
    pass

try:
    from .deployment import (
        KubernetesDeployer,
        AutoScaler,
        LoadBalancer
    )
except ImportError:
    pass

try:
    from .cli import OpenKernelCLI
except ImportError:
    pass

# Export main classes for easy access
__all__ = [
    # Core
    "OpenKernelConfig",
    "ModelArchitecture", 
    "TrainingMode",
    "DatasetType",
    "OptimizationLevel",
    "PrecisionType",
    "TrainingMetrics",
    "InferenceMetrics",
    "SystemMetrics",
    
    # Training (if available)
    "DistributedTrainer",
    "ModelParallelTrainer", 
    "OptimizerFactory",
    "LearningRateScheduler",
    
    # Inference (if available)
    "InferenceEngine",
    "KVCache",
    "SpeculativeDecoder",
    "BatchProcessor",
    
    # Data (if available)
    "DataPipeline",
    "WebCrawler",
    "DataProcessor",
    "QualityFilter",
    
    # CUDA (if available)
    "CUDAKernelGenerator",
    "KernelOptimizer",
    "PerformanceBenchmark",
    
    # Research (if available)
    "ResearchFramework",
    "ExperimentManager",
    "ArchitectureComparator",
    "LongContextEvaluator",
    
    # Monitoring (if available)
    "MetricsCollector",
    "AlertManager",
    "PerformanceProfiler",
    
    # Deployment (if available)
    "KubernetesDeployer",
    "AutoScaler",
    "LoadBalancer",
    
    # CLI (if available)
    "OpenKernelCLI"
]

def get_version():
    """Get OpenKernel version"""
    return __version__

def get_available_modules():
    """Get list of available modules"""
    available = ["core"]
    
    try:
        import openkernel.training
        available.append("training")
    except ImportError:
        pass
        
    try:
        import openkernel.inference
        available.append("inference")
    except ImportError:
        pass
        
    try:
        import openkernel.data
        available.append("data")
    except ImportError:
        pass
        
    try:
        import openkernel.cuda
        available.append("cuda")
    except ImportError:
        pass
        
    try:
        import openkernel.research
        available.append("research")
    except ImportError:
        pass
        
    try:
        import openkernel.monitoring
        available.append("monitoring")
    except ImportError:
        pass
        
    try:
        import openkernel.deployment
        available.append("deployment")
    except ImportError:
        pass
        
    try:
        import openkernel.cli
        available.append("cli")
    except ImportError:
        pass
    
    return available

# Print available modules on import
print(f"OpenKernel v{__version__} loaded successfully")
print(f"Available modules: {get_available_modules()}") 