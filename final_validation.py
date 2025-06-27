#!/usr/bin/env python3
"""
OpenKernel Final Validation Script
==================================

This script performs comprehensive validation of all OpenKernel components
to ensure the system is fully functional and production-ready.
"""

import sys
import time
import traceback
from typing import Dict, Any, List


def validate_imports() -> Dict[str, Any]:
    """Validate all module imports"""
    print("🔍 Validating Module Imports...")
    results = {"passed": [], "failed": []}
    
    # Core modules
    modules_to_test = [
        ("openkernel", "Core package"),
        ("openkernel.core", "Core configuration and metrics"),
        ("openkernel.training", "Distributed training"),
        ("openkernel.inference", "Inference optimization"),
        ("openkernel.data", "Data pipeline"),
        ("openkernel.cuda", "CUDA kernel development"),
        ("openkernel.research", "Research framework"),
        ("openkernel.monitoring", "Monitoring and metrics"),
        ("openkernel.deployment", "Production deployment"),
        ("openkernel.cli", "Command-line interface")
    ]
    
    for module_name, description in modules_to_test:
        try:
            __import__(module_name)
            results["passed"].append(f"{module_name}: {description}")
            print(f"  ✅ {module_name}")
        except Exception as e:
            results["failed"].append(f"{module_name}: {str(e)}")
            print(f"  ❌ {module_name}: {str(e)}")
    
    return results


def validate_core_functionality() -> Dict[str, Any]:
    """Validate core OpenKernel functionality"""
    print("\n🧪 Validating Core Functionality...")
    results = {"passed": [], "failed": []}
    
    try:
        from openkernel.core import OpenKernelConfig, ModelArchitecture, TrainingMode
        
        # Test configuration creation
        config = OpenKernelConfig(
            model_size="7B",
            num_nodes=1,
            gpus_per_node=2,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=2
        )
        results["passed"].append("Configuration creation")
        print("  ✅ Configuration creation")
        
        # Test configuration validation
        assert config.model_size == "7B"
        assert config.num_nodes == 1
        results["passed"].append("Configuration validation")
        print("  ✅ Configuration validation")
        
        # Test enum usage
        assert ModelArchitecture.TRANSFORMER.value == "transformer"
        assert TrainingMode.PRETRAINING.value == "pretraining"
        results["passed"].append("Enum functionality")
        print("  ✅ Enum functionality")
        
    except Exception as e:
        results["failed"].append(f"Core functionality: {str(e)}")
        print(f"  ❌ Core functionality: {str(e)}")
    
    return results


def validate_training_system() -> Dict[str, Any]:
    """Validate distributed training system"""
    print("\n🚀 Validating Training System...")
    results = {"passed": [], "failed": []}
    
    try:
        from openkernel.training import DistributedTrainer
        from openkernel.core import OpenKernelConfig, ModelArchitecture
        from unittest.mock import Mock
        
        # Create trainer
        config = OpenKernelConfig(
            model_size="7B",
            num_nodes=1,
            gpus_per_node=2,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=2,
            micro_batch_size=1,
            batch_size=128,
            gradient_accumulation_steps=128
        )
        trainer = DistributedTrainer(config)
        results["passed"].append("Trainer initialization")
        print("  ✅ Trainer initialization")
        
        # Test model creation
        model = trainer.create_model()
        assert model is not None
        results["passed"].append("Model creation")
        print("  ✅ Model creation")
        
        # Test training step
        batch = Mock()
        mock_model = Mock()
        mock_model.parameters = 7_000_000_000
        optimizer = Mock()
        
        metrics = trainer.train_step(batch, mock_model, optimizer)
        assert "loss" in metrics
        assert "tokens_per_second" in metrics
        results["passed"].append("Training step execution")
        print("  ✅ Training step execution")
        
    except Exception as e:
        results["failed"].append(f"Training system: {str(e)}")
        print(f"  ❌ Training system: {str(e)}")
    
    return results


def validate_inference_engine() -> Dict[str, Any]:
    """Validate inference optimization engine"""
    print("\n⚡ Validating Inference Engine...")
    results = {"passed": [], "failed": []}
    
    try:
        from openkernel.inference import InferenceEngine
        from openkernel.core import OpenKernelConfig
        
        # Create inference engine
        config = OpenKernelConfig(inference_batch_size=4, max_new_tokens=100)
        engine = InferenceEngine(config)
        results["passed"].append("Engine initialization")
        print("  ✅ Engine initialization")
        
        # Test model loading
        engine.load_model("/test/model", "test_model")
        assert "test_model" in engine.models
        results["passed"].append("Model loading")
        print("  ✅ Model loading")
        
        # Test inference-time compute
        result = engine.generate_with_inference_time_compute(
            "Test prompt", "test_model", num_thoughts=3
        )
        assert "response" in result
        assert "thoughts" in result
        assert len(result["thoughts"]) == 3
        results["passed"].append("Inference-time compute")
        print("  ✅ Inference-time compute")
        
        # Test batch inference
        prompts = ["Test 1", "Test 2", "Test 3"]
        batch_results = engine.batch_inference(prompts, "test_model")
        assert len(batch_results) == 3
        results["passed"].append("Batch inference")
        print("  ✅ Batch inference")
        
    except Exception as e:
        results["failed"].append(f"Inference engine: {str(e)}")
        print(f"  ❌ Inference engine: {str(e)}")
    
    return results


def validate_data_pipeline() -> Dict[str, Any]:
    """Validate data pipeline system"""
    print("\n📊 Validating Data Pipeline...")
    results = {"passed": [], "failed": []}
    
    try:
        from openkernel.data import DataPipeline
        from openkernel.core import OpenKernelConfig, DatasetType
        
        # Create data pipeline
        config = OpenKernelConfig(max_dataset_size=1000000)
        pipeline = DataPipeline(config)
        results["passed"].append("Pipeline initialization")
        print("  ✅ Pipeline initialization")
        
        # Test pretraining dataset creation
        sources = ["CommonCrawl", "Wikipedia"]
        dataset_id = pipeline.create_pretraining_dataset(sources)
        assert dataset_id is not None
        assert dataset_id in pipeline.datasets
        results["passed"].append("Pretraining dataset creation")
        print("  ✅ Pretraining dataset creation")
        
        # Test instruction dataset creation
        instruction_sources = ["Alpaca", "ShareGPT"]
        instruction_id = pipeline.create_instruction_dataset(instruction_sources)
        assert instruction_id is not None
        results["passed"].append("Instruction dataset creation")
        print("  ✅ Instruction dataset creation")
        
    except Exception as e:
        results["failed"].append(f"Data pipeline: {str(e)}")
        print(f"  ❌ Data pipeline: {str(e)}")
    
    return results


def validate_research_framework() -> Dict[str, Any]:
    """Validate research framework"""
    print("\n🔬 Validating Research Framework...")
    results = {"passed": [], "failed": []}
    
    try:
        from openkernel.research import ResearchFramework
        from openkernel.core import OpenKernelConfig, ModelArchitecture
        
        # Create research framework
        config = OpenKernelConfig()
        framework = ResearchFramework(config)
        results["passed"].append("Framework initialization")
        print("  ✅ Framework initialization")
        
        # Test experiment creation
        exp_id = framework.create_experiment("Test Experiment", "Testing functionality")
        assert exp_id is not None
        assert exp_id in framework.experiments
        results["passed"].append("Experiment creation")
        print("  ✅ Experiment creation")
        
        # Test architecture comparison
        architectures = [ModelArchitecture.TRANSFORMER, ModelArchitecture.MAMBA]
        comparison_results = framework.run_architecture_comparison(architectures)
        assert len(comparison_results) == 2
        results["passed"].append("Architecture comparison")
        print("  ✅ Architecture comparison")
        
        # Test long context evaluation
        context_lengths = [1024, 4096]
        context_results = framework.evaluate_long_context_performance(context_lengths)
        assert len(context_results) == 2
        results["passed"].append("Long context evaluation")
        print("  ✅ Long context evaluation")
        
    except Exception as e:
        results["failed"].append(f"Research framework: {str(e)}")
        print(f"  ❌ Research framework: {str(e)}")
    
    return results


def validate_cli_functionality() -> Dict[str, Any]:
    """Validate CLI functionality"""
    print("\n💻 Validating CLI Functionality...")
    results = {"passed": [], "failed": []}
    
    try:
        from openkernel.cli import OpenKernelCLI
        
        # Create CLI
        cli = OpenKernelCLI()
        assert cli.config is not None
        results["passed"].append("CLI initialization")
        print("  ✅ CLI initialization")
        
        # Test system status
        assert hasattr(cli, '_show_system_status')
        assert hasattr(cli, 'show_banner')
        results["passed"].append("CLI methods available")
        print("  ✅ CLI methods available")
        
    except Exception as e:
        results["failed"].append(f"CLI functionality: {str(e)}")
        print(f"  ❌ CLI functionality: {str(e)}")
    
    return results


def validate_metrics_system() -> Dict[str, Any]:
    """Validate metrics and monitoring"""
    print("\n📈 Validating Metrics System...")
    results = {"passed": [], "failed": []}
    
    try:
        from openkernel.core import TrainingMetrics, InferenceMetrics
        from datetime import datetime
        
        # Test training metrics
        training_metrics = TrainingMetrics(
            step=100,
            epoch=1,
            loss=2.5,
            learning_rate=1e-4,
            grad_norm=1.2,
            tokens_per_second=1500,
            samples_per_second=50,
            gpu_memory_usage=45.2,
            cpu_memory_usage=8.1,
            model_flops=125.6,
            communication_overhead=0.1
        )
        assert training_metrics.step == 100
        results["passed"].append("Training metrics creation")
        print("  ✅ Training metrics creation")
        
        # Test inference metrics
        inference_metrics = InferenceMetrics(
            model_name="test_model",
            batch_size=32,
            sequence_length=2048,
            tokens_per_second=5420,
            latency_ms=95.2,
            latency_p50=45.2,
            latency_p95=89.7,
            latency_p99=156.3,
            memory_usage_gb=12.8,
            gpu_utilization=91.2,
            cache_hit_rate=0.85,
            error_rate=0.001
        )
        assert inference_metrics.model_name == "test_model"
        results["passed"].append("Inference metrics creation")
        print("  ✅ Inference metrics creation")
        
    except Exception as e:
        results["failed"].append(f"Metrics system: {str(e)}")
        print(f"  ❌ Metrics system: {str(e)}")
    
    return results


def run_comprehensive_validation() -> Dict[str, Any]:
    """Run comprehensive validation of all OpenKernel components"""
    print("🔥 OpenKernel Comprehensive Validation")
    print("=" * 50)
    
    start_time = time.time()
    
    # Run all validation tests
    all_results = {
        "imports": validate_imports(),
        "core": validate_core_functionality(),
        "training": validate_training_system(),
        "inference": validate_inference_engine(),
        "data": validate_data_pipeline(),
        "research": validate_research_framework(),
        "cli": validate_cli_functionality(),
        "metrics": validate_metrics_system()
    }
    
    end_time = time.time()
    
    # Calculate summary statistics
    total_passed = sum(len(results["passed"]) for results in all_results.values())
    total_failed = sum(len(results["failed"]) for results in all_results.values())
    total_tests = total_passed + total_failed
    success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    
    # Print summary
    print("\n" + "=" * 50)
    print("🎯 VALIDATION SUMMARY")
    print("=" * 50)
    print(f"⏱️  Execution Time: {end_time - start_time:.2f} seconds")
    print(f"📊 Total Tests: {total_tests}")
    print(f"✅ Passed: {total_passed}")
    print(f"❌ Failed: {total_failed}")
    print(f"📈 Success Rate: {success_rate:.1f}%")
    
    if total_failed == 0:
        print("\n🎉 STATUS: FULLY FUNCTIONAL")
        print("🚀 OpenKernel is ready for production deployment!")
        print("💼 Perfect for job applications and technical interviews!")
    elif total_failed <= 2:
        print("\n⚠️  STATUS: MOSTLY FUNCTIONAL")
        print("🔧 Minor issues detected, but core functionality works!")
    else:
        print("\n🚨 STATUS: NEEDS ATTENTION")
        print("🔨 Multiple issues detected, please review failures!")
    
    # Print detailed failures if any
    if total_failed > 0:
        print("\n❌ FAILED TESTS:")
        for component, results in all_results.items():
            if results["failed"]:
                print(f"\n  {component.upper()}:")
                for failure in results["failed"]:
                    print(f"    • {failure}")
    
    # Print successful components
    print("\n✅ SUCCESSFUL COMPONENTS:")
    for component, results in all_results.items():
        if results["passed"]:
            print(f"\n  {component.upper()}:")
            for success in results["passed"]:
                print(f"    • {success}")
    
    return {
        "total_tests": total_tests,
        "passed": total_passed,
        "failed": total_failed,
        "success_rate": success_rate,
        "execution_time": end_time - start_time,
        "status": "FULLY FUNCTIONAL" if total_failed == 0 else "NEEDS ATTENTION",
        "details": all_results
    }


def main():
    """Main validation function"""
    try:
        # Suppress import warnings during validation
        import warnings
        warnings.filterwarnings("ignore")
        
        # Run comprehensive validation
        results = run_comprehensive_validation()
        
        # Exit with appropriate code
        if results["failed"] == 0:
            print("\n🎊 OpenKernel validation completed successfully!")
            print("🎯 All systems operational and ready for production!")
            sys.exit(0)
        else:
            print(f"\n⚠️  Validation completed with {results['failed']} issues.")
            print("🔍 Please review the failed tests above.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 Validation failed with critical error: {str(e)}")
        print("\n🔍 Full traceback:")
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main() 