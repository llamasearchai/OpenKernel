# Contributing to OpenKernel

Thank you for your interest in contributing to OpenKernel! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Testing](#testing)
- [Documentation](#documentation)
- [Performance Considerations](#performance-considerations)
- [Security](#security)

## Code of Conduct

This project adheres to a Code of Conduct that we expect all contributors to follow. Please read and follow it in all your interactions with the project.

### Our Standards

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

### Prerequisites

- Python 3.8+ (3.10+ recommended)
- CUDA 11.0+ (for GPU features)
- Git
- Docker (optional, for containerized development)

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/openkernel.git
   cd openkernel
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Development Dependencies**
   ```bash
   pip install -e .[dev,cuda,monitoring]
   ```

4. **Install Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

5. **Verify Installation**
   ```bash
   pytest tests/ -v
   openkernel --help
   ```

### Docker Development

```bash
# Build development image
docker-compose build openkernel-dev

# Run development environment
docker-compose up openkernel-dev

# Run tests in container
docker-compose run openkernel-dev pytest tests/
```

## Contributing Guidelines

### Types of Contributions

We welcome several types of contributions:

- **Bug Reports**: Report issues with clear reproduction steps
- **Feature Requests**: Propose new features with detailed use cases
- **Code Contributions**: Implement bug fixes or new features
- **Documentation**: Improve or add documentation
- **Performance Optimizations**: Enhance CUDA kernels or algorithms
- **Testing**: Add or improve test coverage

### Before You Start

1. **Check Existing Issues**: Look for existing issues or discussions
2. **Create an Issue**: For significant changes, create an issue first
3. **Discuss Approach**: Get feedback on your proposed approach
4. **Check Compatibility**: Ensure changes work across supported platforms

### Code Style

We use automated formatting and linting:

- **Black**: Code formatting (line length: 88)
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **bandit**: Security analysis

Run all checks:
```bash
pre-commit run --all-files
```

### Commit Messages

Follow conventional commit format:

```
type(scope): description

Detailed explanation of changes if needed.

Closes #123
```

Types:
- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions/changes
- `chore`: Build/tooling changes
- `perf`: Performance improvements

### Branch Naming

- `feature/description`: New features
- `fix/description`: Bug fixes
- `docs/description`: Documentation
- `refactor/description`: Code refactoring

## Pull Request Process

### Before Submitting

1. **Update from main**
   ```bash
   git checkout main
   git pull upstream main
   git checkout your-branch
   git rebase main
   ```

2. **Run Tests**
   ```bash
   pytest tests/ -v --cov=openkernel
   ```

3. **Check Code Quality**
   ```bash
   pre-commit run --all-files
   ```

4. **Update Documentation**
   - Update docstrings for new/changed functions
   - Update README if needed
   - Add entries to CHANGELOG.md

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] Added tests for new functionality
- [ ] All tests pass
- [ ] Tested on multiple Python versions
- [ ] Tested with/without CUDA

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
```

### Review Process

1. **Automated Checks**: CI/CD pipeline runs automatically
2. **Code Review**: Maintainers review code quality and design
3. **Testing**: Comprehensive testing on multiple environments
4. **Approval**: At least one maintainer approval required
5. **Merge**: Squash and merge after approval

## Testing

### Test Categories

- **Unit Tests**: Test individual components
- **Integration Tests**: Test component interactions
- **GPU Tests**: Test CUDA kernel functionality
- **Performance Tests**: Benchmark performance
- **Security Tests**: Test for vulnerabilities

### Running Tests

```bash
# All tests
pytest tests/ -v

# Fast tests only
pytest tests/ -m "not slow"

# GPU tests (requires CUDA)
pytest tests/ -m "gpu"

# Integration tests
pytest tests/ -m "integration"

# With coverage
pytest tests/ --cov=openkernel --cov-report=html
```

### Writing Tests

- Use descriptive test names
- Test both success and failure cases
- Mock external dependencies
- Use fixtures for common setup
- Add performance benchmarks for critical paths

Example:
```python
def test_cuda_kernel_generation():
    """Test CUDA kernel generation produces valid code."""
    generator = CUDAKernelGenerator()
    kernel = generator.generate_matmul_kernel(1024, 1024, 1024)
    
    assert kernel.code is not None
    assert "extern \"C\"" in kernel.code
    assert kernel.compile() is True
```

## Documentation

### Types of Documentation

- **API Documentation**: Docstrings for all public functions
- **User Guides**: Step-by-step tutorials
- **Developer Guides**: Architecture and design documents
- **Examples**: Practical usage examples

### Documentation Standards

- Use Google-style docstrings
- Include examples in docstrings
- Keep documentation up-to-date with code
- Use clear, concise language

Example:
```python
def generate_kernel(self, operation: str, size: int) -> CUDAKernel:
    """Generate optimized CUDA kernel for specified operation.
    
    Args:
        operation: Type of operation ('matmul', 'reduce', 'elementwise')
        size: Problem size for optimization
        
    Returns:
        CUDAKernel: Compiled and optimized kernel
        
    Raises:
        ValueError: If operation type is not supported
        
    Example:
        >>> generator = CUDAKernelGenerator()
        >>> kernel = generator.generate_kernel('matmul', 1024)
        >>> result = kernel.execute(a, b)
    """
```

## Performance Considerations

### CUDA Development

- Profile kernels with `nvprof` or `nsight`
- Optimize memory access patterns
- Use appropriate block/grid dimensions
- Leverage shared memory effectively
- Consider warp divergence

### Python Optimization

- Use NumPy for numerical operations
- Minimize Python loops in hot paths
- Profile with `cProfile` or `py-spy`
- Consider Numba for JIT compilation
- Use appropriate data structures

### Memory Management

- Monitor GPU memory usage
- Implement proper cleanup
- Use memory pools when appropriate
- Consider memory-mapped files for large datasets

## Security

### Security Guidelines

- Never commit secrets or credentials
- Validate all inputs
- Use secure random number generation
- Follow OWASP guidelines for web components
- Scan dependencies for vulnerabilities

### Reporting Security Issues

Please report security vulnerabilities privately to security@openkernel.ai rather than creating public issues.

## Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Discord**: Real-time chat (link in README)
- **Email**: security@openkernel.ai for security issues

### Resources

- [Documentation](https://openkernel.readthedocs.io)
- [Examples](./examples/)
- [API Reference](https://openkernel.readthedocs.io/api/)
- [Performance Guide](./docs/performance.md)

## Recognition

Contributors are recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation
- Annual contributor highlights

Thank you for contributing to OpenKernel! Your contributions help make this project better for everyone. 