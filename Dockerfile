# Dockerfile for transformer-spectrum experiments
#
# This Dockerfile creates a CPU-based image for running experiments.
# For GPU support, use the nvidia/cuda base image variant below.
#
# Build:
#   docker build -t transformer-spectrum .
#
# Run training:
#   docker run -v $(pwd)/data:/app/data -v $(pwd)/mlruns:/app/mlruns \
#     transformer-spectrum train -d /app/data/processed/dataset.npy -n my_experiment
#
# Interactive shell:
#   docker run -it --rm transformer-spectrum /bin/bash

# =============================================================================
# CPU Base Image
# =============================================================================
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=42 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# =============================================================================
# Dependencies Stage
# =============================================================================
FROM base as dependencies

# Copy requirements first for caching
COPY requirements.txt pyproject.toml setup.cfg ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# =============================================================================
# Application Stage
# =============================================================================
FROM dependencies as app

# Copy source code
COPY transformer_spectrum/ ./transformer_spectrum/
COPY tests/ ./tests/
COPY Makefile README.md LICENSE pytest.ini ./

# Install package in development mode
RUN pip install -e .

# Create directories for data and outputs
RUN mkdir -p data/raw data/interim data/processed data/external \
    models mlruns reports/figures

# Default command: show help
CMD ["python", "-m", "transformer_spectrum.cli", "--help"]

# =============================================================================
# Entry point for CLI
# =============================================================================
ENTRYPOINT ["python", "-m", "transformer_spectrum.cli"]


# =============================================================================
# GPU Variant (uncomment to use)
# =============================================================================
# To build a GPU-enabled image, replace the base image:
#
# FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 as base
#
# And add CUDA-specific environment variables:
# ENV CUDA_HOME=/usr/local/cuda
# ENV PATH=${CUDA_HOME}/bin:${PATH}
# ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
#
# Then install PyTorch with CUDA support:
# RUN pip install torch --index-url https://download.pytorch.org/whl/cu121
