# ---- Music Generation Docker Image ----
# Build:  docker build -t music-gen .
# Run:    docker run --gpus all -v ./output:/app/output music-gen generate --style "Jazz" --lyrics "..."
# GPU required (CUDA). Use --gpus all or --gpus device=0.

# Use runtime image (smaller than devel — no CUDA compiler toolchain)
FROM pytorch/pytorch:2.10.0-cuda12.6-cudnn8-runtime

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg flac && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (PyTorch already in base image)
COPY pyproject.toml README.md ./
COPY src/ src/

RUN pip install --no-cache-dir -e ".[dev]" && \
    pip install --no-cache-dir "huggingface-hub>=0.36,<1.0" "transformers>=4.57,<5.0" "hf_transfer>=0.1"

# Pre-download models (optional — removes ~15GB but avoids first-run delay)
# Uncomment the next line to include models in the image:
# RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('m-a-p/YuE2-3B'); snapshot_download('m-a-p/YuE2-Vae')"

# Default output directory
RUN mkdir -p /app/output
VOLUME /app/output

# HF cache can be mounted for persistence
ENV HF_HOME=/root/.cache/huggingface

ENTRYPOINT ["music-gen"]
CMD ["--help"]
