#!/usr/bin/env bash
# YuE2 Music Generation - RunPod.io Deployment Script
# Run this on a fresh RunPod PyTorch pod (A100 40GB or better)
#
# Quick start:
#   1. Go to https://runpod.io → Pods → Deploy
#   2. Select "PyTorch" template, GPU: A100 40GB (or RTX 4090)
#   3. Set Container Disk to 50GB+
#   4. Open the pod's Jupyter Lab or Terminal
#   5. Run: curl -sL <raw-url> | bash
#      OR: git clone <repo> && cd music-gen && bash deploy_runpod.sh

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
echo "[1/8] Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq ffmpeg flac > /dev/null 2>&1

# --- Step 2: Find and activate conda ---
echo ""
echo "[2/8] Setting up Python environment..."
# Source conda if available (RunPod images have it but not in PATH)
CONDA_SH=""
for candidate in /root/miniconda3/etc/profile.d/conda.sh /opt/conda/etc/profile.d/conda.sh /home/*/miniconda3/etc/profile.d/conda.sh; do
    if [ -f "$candidate" ]; then
        CONDA_SH="$candidate"
        break
    fi
done

if [ -n "${CONDA_SH}" ]; then
    echo "  Found conda at: ${CONDA_SH}"
    source "${CONDA_SH}"
    if conda env list | grep -q "^${CONDA_ENV} "; then
        echo "  Environment ${CONDA_ENV} already exists, skipping."
    else
        conda create -n "${CONDA_ENV}" python=3.10 -y -q
    fi
    conda activate "${CONDA_ENV}"
else
    echo "  Conda not found, using system Python."
    echo "  Creating venv at ${INSTALL_DIR}/.venv..."
    PYTHON="$(which python3)"
    ${PYTHON} -m venv "${INSTALL_DIR}/.venv"
    source "${INSTALL_DIR}/.venv/bin/activate"
    PYTHON="$(which python)"
    PIP="$(which pip)"
fi

# --- Step 3: Install PyTorch with CUDA ---
echo ""
echo "[3/7] Installing PyTorch with CUDA 12.4..."
cd "${INSTALL_DIR}"
${PIP} install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# --- Step 5: Install project ---
echo ""
echo "[4/7] Installing music-gen package..."
${PIP} install --quiet -e ".[dev]"

# --- Step 6: Install Ollama ---
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

# --- Step 7: Pull default lyrics model ---
echo ""
echo "[6/7] Pulling lyrics model: ${OLLAMA_MODEL} (this may take a while)..."
ollama pull "${OLLAMA_MODEL}"

# --- Step 8: Pre-download YuE2 model weights ---
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
echo "    conda activate ${CONDA_ENV}"
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
