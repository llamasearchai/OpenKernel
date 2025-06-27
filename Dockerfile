# Multi-stage Docker build for OpenKernel
# Stage 1: Base image with CUDA support
FROM nvidia/cuda:11.8-devel-ubuntu20.04 as base

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libssl-dev \
    libffi-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Stage 2: Development image
FROM base as development

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt pyproject.toml ./

# Install Python dependencies
RUN pip install -r requirements.txt
RUN pip install cupy-cuda11x jax[cuda]

# Copy source code
COPY . .

# Install OpenKernel in development mode
RUN pip install -e .[dev,cuda,monitoring]

# Create non-root user
RUN useradd -m -s /bin/bash openkernel
RUN chown -R openkernel:openkernel /app
USER openkernel

# Expose ports for monitoring and API
EXPOSE 8000 8080 6006

CMD ["python", "-m", "openkernel.cli"]

# Stage 3: Production image
FROM base as production

WORKDIR /app

# Copy only necessary files
COPY requirements.txt pyproject.toml ./
COPY openkernel/ ./openkernel/
COPY README.md LICENSE ./

# Install production dependencies only
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir cupy-cuda11x jax[cuda]
RUN pip install --no-cache-dir .

# Create non-root user
RUN useradd -m -s /bin/bash openkernel
RUN chown -R openkernel:openkernel /app
USER openkernel

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import openkernel; print('OpenKernel is healthy')" || exit 1

# Expose ports
EXPOSE 8000

CMD ["openkernel", "--help"]

# Stage 4: Benchmarking image
FROM production as benchmark

USER root
RUN pip install --no-cache-dir pytest-benchmark tensorboard wandb
USER openkernel

CMD ["python", "-m", "pytest", "tests/", "-m", "slow", "--benchmark-json=benchmark.json"] 