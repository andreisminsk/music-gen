# ---- Music Generation Docker Image ----
# Build:  docker build -t pytorch-2.10.0-cuda12.6-music-gen .
# Run:    docker run --gpus all -v ./output:/app/output pytorch-2.10.0-cuda12.6-music-gen generate --style "Jazz" --lyrics "..."
# GPU required (CUDA). Use --gpus all or --gpus device=0.
#
# Base image: RunPod PyTorch with CUDA 12.8 + torch 2.8.0
# We upgrade torch to 2.10.0 (required by yue2_infer).

FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg flac curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade PyTorch to 2.10.0 with CUDA 12.6 (required by yue2_infer)
# cu126 wheels are forward-compatible with CUDA 12.8 runtime
RUN pip install --no-cache-dir \
    torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu126

# Install Python dependencies
COPY pyproject.toml README.md ./
COPY src/ src/

RUN pip install --no-cache-dir -e ".[dev]" && \
    pip install --no-cache-dir \
    "huggingface-hub>=0.36,<1.0" \
    "transformers>=4.57,<5.0" \
    "hf_transfer>=0.1"

# Install Ollama for lyrics generation
RUN curl -fsSL https://ollama.com/install.sh | sh || true

# Pre-download models (optional — adds ~15GB but avoids first-run delay)
# Uncomment to include models in the image:
# RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('m-a-p/YuE2-3B'); snapshot_download('m-a-p/YuE2-Vae')"

# Default output directory
RUN mkdir -p /app/output /app/lyrics
VOLUME /app/output

# HF cache can be mounted for persistence
ENV HF_HOME=/root/.cache/huggingface

ENTRYPOINT ["music-gen"]
CMD ["--help"]
