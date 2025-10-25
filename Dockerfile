# Use a verified Runpod base image with CUDA + Python 3.11
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# ✅ Install git, build tools, ffmpeg (required by audiocraft and torchaudio)
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# ✅ Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ✅ Force reinstall CUDA-compatible PyTorch stack
RUN pip install --no-cache-dir --force-reinstall \
    torch==2.3.1+cu121 \
    torchvision==0.18.1+cu121 \
    torchaudio==2.3.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# ✅ Optional (helps with large deps like audiocraft)
RUN pip install --no-cache-dir setuptools wheel

# Run handler on startup
CMD ["python", "handler.py"]
