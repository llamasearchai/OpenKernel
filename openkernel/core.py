#!/usr/bin/env python3
"""
OpenKernel Core Components
==========================

Core configuration classes, enums, and data structures for the OpenKernel system.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

class ModelArchitecture(Enum):
    """Supported model architectures"""
    TRANSFORMER = "transformer"
    MAMBA = "mamba"
    MIXTURE_OF_EXPERTS = "mixture_of_experts"
    RETNET = "retnet"
    CUSTOM = "custom"

class TrainingMode(Enum):
    """Training modes"""
    PRETRAINING = "pretraining"
    FINETUNING = "finetuning"
    RLHF = "rlhf"
    CONSTITUTIONAL_AI = "constitutional_ai"
    INFERENCE_TIME_COMPUTE = "inference_time_compute"

class DatasetType(Enum):
    """Dataset types"""
    PRETRAINING = "pretraining"
    INSTRUCTION = "instruction"
    REASONING = "reasoning"
    CODE = "code"
    MATH = "math"
    MULTIMODAL = "multimodal"

class OptimizationLevel(Enum):
    """CUDA optimization levels"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXTREME = "extreme"

class PrecisionType(Enum):
    """Precision types for training and inference"""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    MIXED = "mixed"

@dataclass
class OpenKernelConfig:
    """Main configuration for OpenKernel system"""
    
    # Model configuration
    model_architecture: ModelArchitecture = ModelArchitecture.TRANSFORMER
    model_size: str = "7B"  # 7B, 70B, 405B, 1T
    sequence_length: int = 32768
    vocab_size: int = 50257
    hidden_size: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    
    # Training configuration
    training_mode: TrainingMode = TrainingMode.PRETRAINING
    batch_size: int = 1024
    micro_batch_size: int = 8
    gradient_accumulation_steps: int = 128
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    max_steps: int = 100000
    save_interval: int = 1000
    eval_interval: int = 5000
    
    # Distributed training
    num_nodes: int = 8
    gpus_per_node: int = 8
    tensor_parallel_size: int = 8
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 8
    expert_parallel_size: int = 1
    
    # Infrastructure
    master_addr: str = "localhost"
    master_port: int = 29500
    backend: str = "nccl"
    precision: str = "bf16"
    use_tensor_cores: bool = True
    gradient_checkpointing: bool = True
    
    # Data pipeline
    dataset_type: DatasetType = DatasetType.PRETRAINING
    num_workers: int = 16
    prefetch_factor: int = 4
    max_dataset_size: int = 1_000_000_000
    quality_threshold: float = 0.8
    dedup_threshold: float = 0.9
    
    # Inference optimization
    inference_batch_size: int = 32
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    use_kv_cache: bool = True
    use_speculative_decoding: bool = True
    speculation_lookahead: int = 4
    
    # CUDA kernel optimization
    optimization_level: OptimizationLevel = OptimizationLevel.ADVANCED
    target_architecture: str = "A100"  # A100, H100, RTX4090
    use_custom_kernels: bool = True
    kernel_cache_dir: str = "./cuda_cache"
    
    # Monitoring and logging
    log_level: str = "INFO"
    wandb_project: str = "openkernel"
    wandb_entity: str = "ai-research"
    enable_profiling: bool = True
    profile_interval: int = 100
    
    # Production deployment
    serve_port: int = 8000
    max_concurrent_requests: int = 1000
    request_timeout: int = 30
    health_check_interval: int = 30
    auto_scaling_enabled: bool = True
    min_replicas: int = 1
    max_replicas: int = 100
    
    def __post_init__(self):
        """Post-initialization setup"""
        # Create necessary directories
        directories = [
            "./checkpoints", "./logs", "./data", "./results", 
            "./cache", "./cuda_cache", "./experiments", "./models"
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Validate configuration
        self._validate_config()
        
        # Setup logging
        self._setup_logging()
    
    def _validate_config(self):
        """Validate configuration parameters"""
        # Model size validation
        valid_sizes = ["7B", "70B", "405B", "1T"]
        if self.model_size not in valid_sizes:
            raise ValueError(f"Invalid model size: {self.model_size}. Must be one of {valid_sizes}")
        
        # Distributed training validation
        total_gpus = self.num_nodes * self.gpus_per_node
        tp_pp_dp = self.tensor_parallel_size * self.pipeline_parallel_size * self.data_parallel_size
        if tp_pp_dp > total_gpus:
            raise ValueError(f"Parallelism configuration requires {tp_pp_dp} GPUs but only {total_gpus} available")
        
        # Batch size validation
        if self.batch_size % (self.micro_batch_size * self.gradient_accumulation_steps) != 0:
            raise ValueError("batch_size must be divisible by (micro_batch_size * gradient_accumulation_steps)")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('./logs/openkernel.log'),
                logging.StreamHandler()
            ]
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        config_dict = {}
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, Enum):
                config_dict[field_name] = field_value.value
            elif isinstance(field_value, datetime):
                config_dict[field_name] = field_value.isoformat()
            else:
                config_dict[field_name] = field_value
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "OpenKernelConfig":
        """Create configuration from dictionary"""
        # Convert enum strings back to enums
        if "model_architecture" in config_dict:
            config_dict["model_architecture"] = ModelArchitecture(config_dict["model_architecture"])
        if "training_mode" in config_dict:
            config_dict["training_mode"] = TrainingMode(config_dict["training_mode"])
        if "dataset_type" in config_dict:
            config_dict["dataset_type"] = DatasetType(config_dict["dataset_type"])
        if "optimization_level" in config_dict:
            config_dict["optimization_level"] = OptimizationLevel(config_dict["optimization_level"])
        
        return cls(**config_dict)
    
    def save(self, filepath: str):
        """Save configuration to file"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> "OpenKernelConfig":
        """Load configuration from file"""
        import json
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

@dataclass
class TrainingMetrics:
    """Training metrics tracking"""
    step: int
    epoch: int
    loss: float
    learning_rate: float
    grad_norm: float
    tokens_per_second: float
    samples_per_second: float
    gpu_memory_usage: float
    cpu_memory_usage: float
    model_flops: float
    communication_overhead: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            "step": self.step,
            "epoch": self.epoch,
            "loss": self.loss,
            "learning_rate": self.learning_rate,
            "grad_norm": self.grad_norm,
            "tokens_per_second": self.tokens_per_second,
            "samples_per_second": self.samples_per_second,
            "gpu_memory_usage": self.gpu_memory_usage,
            "cpu_memory_usage": self.cpu_memory_usage,
            "model_flops": self.model_flops,
            "communication_overhead": self.communication_overhead,
            "timestamp": self.timestamp.isoformat()
        }

@dataclass
class InferenceMetrics:
    """Inference performance metrics"""
    model_name: str
    batch_size: int
    sequence_length: int
    tokens_per_second: float
    latency_ms: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    memory_usage_gb: float
    gpu_utilization: float
    cache_hit_rate: float
    error_rate: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "tokens_per_second": self.tokens_per_second,
            "latency_ms": self.latency_ms,
            "latency_p50": self.latency_p50,
            "latency_p95": self.latency_p95,
            "latency_p99": self.latency_p99,
            "memory_usage_gb": self.memory_usage_gb,
            "gpu_utilization": self.gpu_utilization,
            "cache_hit_rate": self.cache_hit_rate,
            "error_rate": self.error_rate,
            "timestamp": self.timestamp.isoformat()
        }

@dataclass
class SystemMetrics:
    """System-level metrics"""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: float
    gpu_usage: List[float]
    gpu_memory: List[float]
    temperature: List[float]
    power_usage: List[float]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for monitoring"""
        return {
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "disk_usage": self.disk_usage,
            "network_io": self.network_io,
            "gpu_usage": self.gpu_usage,
            "gpu_memory": self.gpu_memory,
            "temperature": self.temperature,
            "power_usage": self.power_usage,
            "timestamp": self.timestamp.isoformat()
        } 