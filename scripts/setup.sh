#!/bin/bash

# OpenKernel Development Environment Setup Script
# Supports Ubuntu/Debian, CentOS/RHEL, and macOS

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -f /etc/debian_version ]; then
            OS="debian"
        elif [ -f /etc/redhat-release ]; then
            OS="redhat"
        else
            OS="linux"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    else
        log_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
    log_info "Detected OS: $OS"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Install system dependencies
install_system_deps() {
    log_info "Installing system dependencies..."
    
    case $OS in
        "debian")
            sudo apt-get update
            sudo apt-get install -y \
                build-essential \
                cmake \
                git \
                wget \
                curl \
                python3 \
                python3-dev \
                python3-pip \
                python3-venv \
                libssl-dev \
                libffi-dev \
                libbz2-dev \
                libreadline-dev \
                libsqlite3-dev \
                pkg-config
            ;;
        "redhat")
            sudo yum groupinstall -y "Development Tools"
            sudo yum install -y \
                cmake \
                git \
                wget \
                curl \
                python3 \
                python3-devel \
                python3-pip \
                openssl-devel \
                libffi-devel \
                bzip2-devel \
                readline-devel \
                sqlite-devel \
                pkgconfig
            ;;
        "macos")
            if ! command_exists brew; then
                log_info "Installing Homebrew..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            fi
            brew install cmake git wget curl python@3.10
            ;;
    esac
    
    log_success "System dependencies installed"
}

# Install CUDA (Linux only)
install_cuda() {
    if [[ "$OS" == "macos" ]]; then
        log_warning "CUDA not available on macOS, skipping..."
        return
    fi
    
    if command_exists nvcc; then
        log_info "CUDA already installed: $(nvcc --version | grep release)"
        return
    fi
    
    log_info "Installing CUDA toolkit..."
    
    case $OS in
        "debian")
            wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
            sudo dpkg -i cuda-keyring_1.0-1_all.deb
            sudo apt-get update
            sudo apt-get -y install cuda-toolkit-11-8
            rm cuda-keyring_1.0-1_all.deb
            ;;
        "redhat")
            sudo yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
            sudo yum clean all
            sudo yum -y install cuda-toolkit-11-8
            ;;
    esac
    
    # Add CUDA to PATH
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    
    log_success "CUDA toolkit installed"
    log_warning "Please restart your shell or run 'source ~/.bashrc' to update PATH"
}

# Install Docker
install_docker() {
    if command_exists docker; then
        log_info "Docker already installed: $(docker --version)"
        return
    fi
    
    log_info "Installing Docker..."
    
    case $OS in
        "debian")
            curl -fsSL https://get.docker.com -o get-docker.sh
            sudo sh get-docker.sh
            sudo usermod -aG docker $USER
            rm get-docker.sh
            ;;
        "redhat")
            sudo yum install -y yum-utils
            sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
            sudo yum install -y docker-ce docker-ce-cli containerd.io
            sudo systemctl start docker
            sudo systemctl enable docker
            sudo usermod -aG docker $USER
            ;;
        "macos")
            log_warning "Please install Docker Desktop for Mac manually from https://docker.com/products/docker-desktop"
            return
            ;;
    esac
    
    # Install Docker Compose
    sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    
    log_success "Docker installed"
    log_warning "Please log out and back in to use Docker without sudo"
}

# Setup Python environment
setup_python_env() {
    log_info "Setting up Python environment..."
    
    # Check Python version
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    REQUIRED_VERSION="3.8"
    
    if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
        log_error "Python $REQUIRED_VERSION or higher required, found $PYTHON_VERSION"
        exit 1
    fi
    
    # Create virtual environment
    if [ ! -d ".venv" ]; then
        log_info "Creating virtual environment..."
        python3 -m venv .venv
    fi
    
    # Activate virtual environment
    source .venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    log_success "Python environment ready"
}

# Install Python dependencies
install_python_deps() {
    log_info "Installing Python dependencies..."
    
    source .venv/bin/activate
    
    # Install basic dependencies
    pip install -r requirements.txt
    
    # Install CUDA dependencies if available
    if command_exists nvcc && [[ "$OS" != "macos" ]]; then
        log_info "Installing CUDA Python packages..."
        pip install cupy-cuda11x jax[cuda]
    else
        log_warning "CUDA not available, installing CPU-only packages..."
        pip install jax[cpu]
    fi
    
    # Install development dependencies
    pip install -e .[dev,monitoring]
    
    log_success "Python dependencies installed"
}

# Setup pre-commit hooks
setup_pre_commit() {
    log_info "Setting up pre-commit hooks..."
    
    source .venv/bin/activate
    
    if [ -f ".pre-commit-config.yaml" ]; then
        pre-commit install
        pre-commit install --hook-type commit-msg
        log_success "Pre-commit hooks installed"
    else
        log_warning "No pre-commit config found, skipping..."
    fi
}

# Run tests
run_tests() {
    log_info "Running test suite..."
    
    source .venv/bin/activate
    
    # Run fast tests only during setup
    pytest tests/ -v -m "not slow and not gpu" --tb=short
    
    if [ $? -eq 0 ]; then
        log_success "All tests passed"
    else
        log_warning "Some tests failed, but setup continues..."
    fi
}

# Setup monitoring (optional)
setup_monitoring() {
    read -p "Do you want to set up monitoring stack? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Setting up monitoring stack..."
        
        # Create monitoring directories
        mkdir -p monitoring/grafana/{dashboards,provisioning}
        mkdir -p monitoring/prometheus
        
        # Start monitoring stack
        docker-compose --profile monitoring up -d
        
        log_success "Monitoring stack started"
        log_info "Grafana: http://localhost:3000 (admin/openkernel123)"
        log_info "Prometheus: http://localhost:9090"
    fi
}

# Verify installation
verify_installation() {
    log_info "Verifying installation..."
    
    source .venv/bin/activate
    
    # Test imports
    python -c "import openkernel; print(f'OpenKernel version: {openkernel.__version__}')" || {
        log_error "Failed to import OpenKernel"
        exit 1
    }
    
    # Test CLI
    openkernel --help > /dev/null || {
        log_error "OpenKernel CLI not working"
        exit 1
    }
    
    # Check CUDA availability
    if python -c "import cupy; print('CUDA available')" 2>/dev/null; then
        log_success "CUDA support verified"
    else
        log_info "CUDA support not available (CPU-only mode)"
    fi
    
    log_success "Installation verified successfully"
}

# Print final instructions
print_instructions() {
    echo
    echo "=================================="
    echo "   OpenKernel Setup Complete!    "
    echo "=================================="
    echo
    echo "To get started:"
    echo "1. Activate the virtual environment:"
    echo "   source .venv/bin/activate"
    echo
    echo "2. Run the demo:"
    echo "   python demo_script.py"
    echo
    echo "3. Or use the CLI:"
    echo "   openkernel --help"
    echo
    echo "4. Run tests:"
    echo "   pytest tests/ -v"
    echo
    echo "5. Start development with Docker:"
    echo "   docker-compose up openkernel-dev"
    echo
    echo "Documentation: https://openkernel.readthedocs.io"
    echo "Issues: https://github.com/openkernel/openkernel/issues"
    echo
}

# Main execution
main() {
    echo "OpenKernel Development Environment Setup"
    echo "========================================"
    echo
    
    detect_os
    
    # Check if running as root
    if [ "$EUID" -eq 0 ]; then
        log_error "Do not run this script as root"
        exit 1
    fi
    
    # Install components
    install_system_deps
    install_cuda
    install_docker
    setup_python_env
    install_python_deps
    setup_pre_commit
    run_tests
    setup_monitoring
    verify_installation
    print_instructions
}

# Handle script arguments
case "${1:-}" in
    "--help"|"-h")
        echo "Usage: $0 [options]"
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --cuda-only    Install only CUDA toolkit"
        echo "  --docker-only  Install only Docker"
        echo "  --python-only  Setup only Python environment"
        exit 0
        ;;
    "--cuda-only")
        detect_os
        install_cuda
        exit 0
        ;;
    "--docker-only")
        detect_os
        install_docker
        exit 0
        ;;
    "--python-only")
        setup_python_env
        install_python_deps
        setup_pre_commit
        verify_installation
        exit 0
        ;;
    "")
        main
        ;;
    *)
        log_error "Unknown option: $1"
        echo "Use --help for usage information"
        exit 1
        ;;
esac 