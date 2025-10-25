# Use a verified, modern Runpod image with Python 3.11 and a compatible CUDA version
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set the working directory
WORKDIR /app

# ✅ INSTALL GIT FIRST - BEFORE COPY AND PIP
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy your project files (AFTER installing git)
COPY . /app

# 1. Install the dependencies (git is now available)
RUN pip install --no-cache-dir -r requirements.txt

# 2. Force the re-installation of a known-good, CUDA-compatible PyTorch stack.
RUN pip install --no-cache-dir --force-reinstall torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Command to run your handler script when the worker starts
CMD ["python", "handler.py"]
