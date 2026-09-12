# ---- Music Generation Docker Image ----
# Build:  docker build -t ghcr.io/andreisminsk/music-gen:0.1.0-torch2.10.0-cu128 .
# Run:    docker run --gpus all -v ./output:/app/output ghcr.io/andreisminsk/music-gen:0.1.0-torch2.10.0-cu128 generate --style "Jazz" --lyrics "..."
# GPU required (CUDA). Use --gpus all or --gpus device=0.
#
# Base image: RunPod PyTorch with CUDA 12.8 + torch 2.8.0
# We upgrade torch to 2.10.0 (required by yue2_infer) with CUDA 12.8 wheels
# to support Blackwell GPUs (sm_120). The base image's CUDA 12.8 runtime
# is fully compatible with the cu128 PyTorch wheels.

FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg flac curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade PyTorch to 2.10.0 with CUDA 12.8 (required by yue2_infer)
# cu128 wheels support Blackwell GPUs (sm_120)
RUN pip install --no-cache-dir \
    torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu128

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

# Start Ollama in background, then keep container alive
# RunPod users SSH in and run: music-gen generate ...
COPY <<'EOF' /app/entrypoint.sh
#!/bin/bash
set -e

# Start Ollama if available
if command -v ollama &> /dev/null; then
    ollama serve &>/dev/null &
    sleep 2
fi

# Keep container alive
exec "$@"
EOF
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["sleep", "infinity"]
