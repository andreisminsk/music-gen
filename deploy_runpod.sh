#!/usr/bin/env bash
# YuE2 Music Generation - RunPod.io Deployment Script
# Run this on a fresh RunPod PyTorch pod (A100 40GB or better)
#
# Quick start:
#   1. Go to https://runpod.io → Pods → Deploy
#   2. Select "PyTorch" template, GPU: A100 40GB (or RTX 4090)
#   3. Set Container Disk to 50GB+
#   4. Open the pod's Jupyter Lab or Terminal
#   5. Clone the repo, cd into it, and run: bash deploy_runpod.sh

set -euo pipefail

echo "=========================================="
echo "  YuE2 Music Generation - RunPod Setup"
echo "=========================================="

# --- Config ---
# Auto-detect: use the directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${INSTALL_DIR:-${SCRIPT_DIR}}"
PYTHON="${PYTHON:-python3}"
PIP="${PIP:-pip3}"
CONDA_ENV="${CONDA_ENV:-yue2}"
OLLAMA_MODEL="${OLLAMA_MODEL:-gemma4:31b-cloud}"

# --- Step 1: System deps ---
echo ""
echo "[1/7] Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq ffmpeg flac > /dev/null 2>&1

# --- Step 2: Find and activate conda, or fall back to venv ---
echo ""
echo "[2/7] Setting up Python environment..."
cd "${INSTALL_DIR}"

CONDA_SH=""
for candidate in /opt/conda/etc/profile.d/conda.sh /root/miniconda3/etc/profile.d/conda.sh /root/anaconda3/etc/profile.d/conda.sh /home/*/miniconda3/etc/profile.d/conda.sh /home/*/anaconda3/etc/profile.d/conda.sh; do
    if [ -f "$candidate" ]; then
        CONDA_SH="$candidate"
        break
    fi
done

USE_CONDA=false
if [ -n "${CONDA_SH}" ]; then
    echo "  Found conda at: ${CONDA_SH}"
    source "${CONDA_SH}"
    if conda env list | grep -q "^${CONDA_ENV} "; then
        echo "  Environment ${CONDA_ENV} already exists, skipping."
    else
        conda create -n "${CONDA_ENV}" python=3.10 -y -q
    fi
    conda activate "${CONDA_ENV}"
    USE_CONDA=true
else
    echo "  Conda not found, using venv."
    ${PYTHON} -m venv "${INSTALL_DIR}/.venv"
    source "${INSTALL_DIR}/.venv/bin/activate"
    PYTHON="$(which python)"
    PIP="$(which pip)"
fi
# --- Step 3: Install PyTorch 2.10 with CUDA FIRST (avoids slow PyPI download) ---
echo ""
echo "[3/7] Installing PyTorch 2.10 with CUDA 12.6..."
${PIP} install --quiet torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu126

# --- Step 4: Install project (torch already satisfied, fast) ---
echo ""
echo "[4/7] Installing music-gen package..."
cd "${INSTALL_DIR}"
${PIP} install --quiet -e ".[dev]"

# Pin huggingface-hub to compatible version (yue2_infer and transformers require <1.0)
echo "  Pinning huggingface-hub>=0.36,<1.0..."
${PIP} install --quiet "huggingface-hub>=0.36,<1.0"

# --- Step 5: Install Ollama ---
echo ""
echo "[5/7] Installing Ollama for lyrics generation..."
if command -v ollama &> /dev/null; then
    echo "  Ollama already installed, skipping."
else
    curl -fsSL https://ollama.com/install.sh | sh
    echo "  Starting Ollama server..."
    ollama serve &> /dev/null &
    sleep 5
    echo "  Ollama installed and started."
fi

# Ensure Ollama is running
if ! pgrep -x ollama &> /dev/null; then
    echo "  Starting Ollama server..."
    ollama serve &> /dev/null &
    sleep 3
fi

# --- Step 6: Pull default lyrics model ---
echo ""
echo "[6/7] Pulling lyrics model: ${OLLAMA_MODEL} (this may take a while)..."
ollama pull "${OLLAMA_MODEL}"

# --- Step 7: Pre-download YuE2 model weights ---
echo ""
echo "[7/7] Pre-downloading YuE2-3B model weights (this takes a while)..."
${PYTHON} -c "
from huggingface_hub import snapshot_download
print('Downloading YuE2-3B...')
snapshot_download('m-a-p/YuE2-3B')
print('Downloading YuE2-Vae...')
snapshot_download('m-a-p/YuE2-Vae')
print('All models downloaded.')
"

echo ""
echo "=========================================="
echo "  ✅ Setup complete!"
echo "=========================================="
echo ""
echo "  Activate the environment:"
if [ "${USE_CONDA}" = true ]; then
    echo "    conda activate ${CONDA_ENV}"
else
    echo "    source ${INSTALL_DIR}/.venv/bin/activate"
fi
echo ""
echo "  Generate lyrics with a local LLM:"
echo "    music-gen lyrics-gen \\"
echo "      --style 'Indie folk rock, warm acoustic guitar, reflective male vocal' \\"
echo "      --topic 'Old programmer flying between two countries' \\"
echo "      --language English \\"
echo "      --output lyrics/song.txt"
echo ""
echo "  Generate a song from a lyrics file:"
echo "    music-gen generate \\"
echo "      --style 'Indie folk rock, warm acoustic guitar, reflective male vocal' \\"
echo "      --lyrics-file lyrics/song.txt \\"
echo "      --seed 42"
echo ""
echo "  Generate a song from inline lyrics:"
echo "    music-gen generate \\"
echo "      --style 'Jazz, warm vocal, piano, upright bass' \\"
echo "      --lyrics '[Verse 1]\nWalking down the avenue\n\n[Chorus]\nTonight we break the chain' \\"
echo "      --seed 42"
echo ""
echo "  Use a different Ollama model for lyrics:"
echo "    music-gen lyrics-gen --style 'Rock' --model qwen3.5:35b-mlx --output lyrics/song.txt"
echo ""
echo "  Or use the Python API:"
echo "    python -m music_gen.examples  # runs the russian_rock example"
echo ""
echo "  Output files will be in: ${INSTALL_DIR}/output/"
echo ""
echo "  Installed models:"
echo "    YuE2:     m-a-p/YuE2-3B + m-a-p/YuE2-Vae"
echo "    Ollama:   ${OLLAMA_MODEL}"
echo "=========================================="
