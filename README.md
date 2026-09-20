# 🎵 music-gen

YuE2-3B music generation app — turn lyrics + style into complete songs with vocals and accompaniment.

## Requirements

- Python 3.10+
- NVIDIA GPU with 24GB VRAM (BF16 support)
- CUDA

## Setup

```bash
pip install -e .
```

The `yue2_infer` package will be downloaded and installed automatically on first run.

## Usage

### Generate a song

```bash
# From CLI arguments
music-gen generate \
  --style "Jazz, warm vocal, piano, upright bass" \
  --lyrics "[Verse 1]\nWalking down the avenue\n\n[Chorus]\nTonight we break the chain" \
  --seed 42

# From a lyrics text file
music-gen generate \
  --style "Jazz, warm vocal, piano, upright bass" \
  --lyrics-file lyrics/my_song.txt \
  --seed 42

# From a prompt JSON file
music-gen generate --prompt-json examples/tonight-awake.json

# With options
music-gen generate \
  --style "Cyber metal, aggressive vocals" \
  --lyrics "[Verse 1]\nSteel and silicon collide\n\n[Chorus]\nWe are the machine" \
  --cot full \
  --cfg-scale 1.2 \
  --seed 123 \
  --output-dir output \
  --filename my_song
```

### Export a plan (ABC notation only, no audio)

```bash
# Using the plan command
music-gen plan \
  --style "Jazz-funk, warm vocal" \
  --lyrics "[Verse 1]\nSome lyrics here" \
  --seed 42

# Or using --melody-only (same result, skips audio generation)
music-gen generate \
  --style "Jazz-funk, warm vocal" \
  --lyrics "[Verse 1]\nSome lyrics here" \
  --cot melody --melody-only --seed 42
```

### Command reference

#### `music-gen generate`

Generate a complete song with vocals and accompaniment from a style prompt and lyrics. Supports inline lyrics, lyrics files, or prompt JSON files. Outputs FLAC audio plus artifacts (ABC score, tokens, latents).

| Flag | Default | Description |
|------|---------|-------------|
| `--style`, `-s` | *(required)* | Style/genre prompt |
| `--lyrics`, `-l` | — | Lyrics text (use `\n` for line breaks) |
| `--lyrics-file`, `-L` | — | Path to a lyrics text file |
| `--prompt-json`, `-p` | — | Path to a prompt JSON file (keys: style, lyrics, seed) |
| `--cot` | `full` | Chain-of-thought mode: `full`, `melody`, or `off` |
| `--cfg-scale` | `1.2` | Classifier-free guidance scale |
| `--seed` | — | Random seed for reproducibility |
| `--abc` | — | Path to ABC notation file (for covers/edits) |
| `--output-dir`, `-o` | `output` | Output directory |
| `--filename`, `-f` | `song` | Output filename (no extension) |
| `--model` | `m-a-p/YuE2-3B` | Model repo ID |
| `--vae` | `m-a-p/YuE2-Vae` | VAE variant (`YuE2-Vae` or `YuE2-Vae-legacy`) |
| `--device` | `cuda` | Torch device |
| `--no-progress` | — | Suppress progress messages |
| `--generate-lyrics` | — | Generate lyrics via Ollama before composing |
| `--topic`, `-t` | — | Topic for lyrics generation (with `--generate-lyrics`) |
| `--language` | `English` | Language for lyrics generation |
| `--lyrics-model` | `gemma4:31b-cloud` | Ollama model for lyrics generation |
| `--lyrics-output` | `lyrics/generated.txt` | Save generated lyrics to this file |

#### `music-gen plan`

Export a symbolic music plan (ABC notation) without generating audio. Use this to preview or edit the melody and chords before generating a full song. The plan can be edited and fed back into `generate` with `--abc`.

| Flag | Default | Description |
|------|---------|-------------|
| `--style`, `-s` | *(required)* | Style/genre prompt |
| `--lyrics`, `-l` | *(required)* | Lyrics text |
| `--cot` | `full` | `full` or `melody` |
| `--seed` | — | Random seed |
| `--output-dir`, `-o` | `output` | Output directory |
| `--model` | `m-a-p/YuE2-3B` | Model repo ID |
| `--vae` | `m-a-p/YuE2-Vae` | VAE variant |
| `--device` | `cuda` | Torch device |

#### `music-gen lyrics-gen`

Generate song lyrics using a local LLM via Ollama. Produces properly structured lyrics with `[Verse]`, `[Chorus]`, `[Bridge]` markers that YuE2 expects. Supports any language and genre.

| Flag | Default | Description |
|------|---------|-------------|
| `--style`, `-s` | *(required)* | Style/genre prompt |
| `--topic`, `-t` | — | Optional topic or story |
| `--language`, `-L` | `English` | Language to write in |
| `--model`, `-m` | `gemma4:31b-cloud` | Ollama model for lyrics generation |
| `--sections`, `-n` | `4` | Approximate number of sections |
| `--temperature` | `0.8` | Sampling temperature |
| `--output`, `-o` | `lyrics/song.txt` | Output file path |
| `--print-only` | — | Print to stdout, don't save |

### Chain-of-thought modes

| Mode | Flag | Description |
|------|------|-------------|
| Full | `--cot full` | Melody + chord planning (default, best quality) |
| Melody | `--cot melody` | Melody-only planning (recommended for covers) |
| Off | `--cot off` | No symbolic plan, direct generation |

### VAE variants

YuE2 uses a separate VAE (Variational Autoencoder) model to decode acoustic latents into stereo audio. Two variants are available:

- **`m-a-p/YuE2-Vae`** (default) — delivers better perceptual audio quality with cleaner highs and more natural timbre. Recommended for listening.
- **`m-a-p/YuE2-Vae-legacy`** — achieves higher scores on the WildSongBench benchmark (used in the paper's reported results). Use this if you want to reproduce benchmark numbers.

```bash
# Use legacy VAE for benchmark reproduction
music-gen generate --vae m-a-p/YuE2-Vae-legacy ...
```

### Python API

```python
from music_gen.pipeline import load_pipeline, generate_song

pipe = load_pipeline()
result = generate_song(
    pipe,
    style="Jazz, warm vocal, piano, upright bass",
    lyrics="[Verse 1]\nWalking down the avenue\n\n[Chorus]\nTonight we break the chain",
    cot="full",
    seed=42,
)
print(f"Saved to: {result}")
pipe.close()
```

### Generate lyrics with a local LLM

YuE2 doesn't generate lyrics — it takes them as input. Use `lyrics-gen` to create
properly structured lyrics with `[Verse]`, `[Chorus]`, etc. markers using an Ollama model.

```bash
# Generate Russian rock lyrics
music-gen lyrics-gen --style "Russian rock, powerful male vocal" --language Russian

# With a topic
music-gen lyrics-gen --style "Jazz ballad" --language English --topic "rainy night in the city"

# Save to a custom path
music-gen lyrics-gen --style "Cyber metal" --output lyrics/my_song.txt

# Print to stdout only (no file)
music-gen lyrics-gen --style "Disco funk" --print-only

# Use a different Ollama model
music-gen lyrics-gen --style "Folk" --model qwen3.5:35b-mlx
```

### End-to-end: generate lyrics → compose music

```bash
# One command: generate lyrics then compose
music-gen generate \
  --style "Russian rock, powerful male vocal" \
  --generate-lyrics --language Russian --topic "train journey" \
  --seed 2026

# Or as separate steps:
music-gen lyrics-gen --style "Russian rock" --language Russian --output lyrics/song.txt
music-gen generate --style "Russian rock, powerful male vocal" --lyrics-file lyrics/song.txt --seed 42
```

### Example: The Programmer's Song

```bash
# Generate lyrics about an old programmer flying between two countries
music-gen lyrics-gen \
  --style "Indie folk rock, warm acoustic guitar, reflective male vocal" \
  --topic "Old programmer flying between two countries, enjoying family time in home country, enjoying mountains, sun and sea in the other country" \
  --language English \
  --output lyrics/programmer_song.txt
```

Output (`lyrics/programmer_song.txt`):

```
[Verse 1]
Silver wings cutting through a velvet haze
Trading syntax for the golden autumn blaze
I left the humming servers and the flickering screen
To find the quiet places where the air is clean

[Chorus]
Between two horizons, I'm drifting in flight
Chasing the morning, escaping the night
From the warmth of the hearth to the salt of the spray
I'm living the rhythm of a long-distance day

[Verse 2]
Laughter in the kitchen, a child's sudden hand
Roots digging deeper in my father's native land
Then I wake to the peaks where the granite meets blue
And the Mediterranean sun makes the world feel new

[Chorus]
Between two horizons, I'm drifting in flight
Chasing the morning, escaping the night
From the warmth of the hearth to the salt of the spray
I'm living the rhythm of a long-distance day

[Bridge]
My mind is a map of a thousand old lines
But my heart is a forest of cedar and pines
No logic can measure the pull of the tide
Or the peace that I feel with my kin by my side

[Outro]
Just a ghost in the clouds, a soul on the wing
Listening to the song that the trade winds sing
Mountains and oceans, home and the sea
Finally finding where I'm meant to be
```

Then compose music from the lyrics file:

```bash
music-gen generate \
  --style "Indie folk rock, warm acoustic guitar, reflective male vocal" \
  --lyrics-file lyrics/programmer_song.txt \
  --seed 42
```

### Built-in example prompts

The `examples.py` module ships with ready-made prompts for quick testing:

```bash
# Generate lyrics in different styles
music-gen lyrics-gen --style "Jazz-funk, warm lead vocal, Rhodes piano" --language English --print-only
music-gen lyrics-gen --style "Cyber metal, aggressive vocals, industrial" --language English --print-only
music-gen lyrics-gen --style "Mandarin pop, gentle vocal, acoustic guitar" --language Chinese --print-only
music-gen lyrics-gen --style "Russian rock, powerful male vocal" --language Russian --print-only
```

Or use them from Python:

```python
from music_gen.examples import get_example, list_examples

print(list_examples())  # ['jazz_funk', 'cyber_metal', 'mandarin_pop', 'russian_rock']

prompt = get_example("russian_rock")
# {"style": "Russian rock, ...", "lyrics": "...", "seed": 2026}
```

## ☁️ Deploying on RunPod.io

YuE2 requires an NVIDIA GPU with 24GB+ VRAM. RunPod is the easiest way to get started.

### 1. Create a Pod

1. Go to [runpod.io](https://runpod.io) → **Pods** → **Deploy**
2. **Template:** Select **PyTorch** (comes with CUDA + Python)
3. **GPU:** Choose one of:
   - **RTX 4090** — 24GB VRAM, ~$0.44/hr (minimum viable)
   - **A100 40GB** — ~$1.14/hr (recommended, faster)
   - **A100 80GB** — ~$1.64/hr (best for concurrent serving)
4. **Container Disk:** Set to **50GB** (model weights are ~7GB + PyTorch)
5. Click **Deploy**

### 2. Open Terminal

Once the pod is running, click **Connect** → **Start Terminal** (or use Jupyter Lab).

### 3. Set up HuggingFace token (recommended)

YuE2 model weights are ~7GB. Without a token, you may hit rate limits. A free HF account gives you 5x faster downloads.

1. Create a free account at https://huggingface.co/join
2. Go to https://huggingface.co/settings/tokens → **Create token** (Read access is enough)
3. On RunPod, log in:
   ```bash
   huggingface-cli login
   # Paste your token when prompted
   ```

Or pass it as an environment variable:
```bash
HF_TOKEN=hf_xxxxx bash deploy_runpod.sh
```

### 4. Run the Setup Script

```bash
cd /workspace/music-gen
bash deploy_runpod.sh
```

The script will:
- Install ffmpeg/flac
- Set up a Python environment (conda or venv)
- Install the `music-gen` package
- Install PyTorch 2.10 with CUDA 12.6
- Pin `huggingface-hub<1.0` for compatibility
- Install Ollama and pull the lyrics model
- Pre-download all YuE2 model weights (~7GB)

### 4. Generate Music

```bash
# Activate environment (conda or venv — the script tells you which)
```bash
conda activate yue2
# OR: source /workspace/music-gen/.venv/bin/activate

# Generate lyrics then compose in one command
music-gen generate \
  --style "Russian rock, powerful male vocal" \
  --generate-lyrics --language Russian --topic "train journey" \
  --seed 2026

# From a lyrics file
music-gen generate \
  --style "Jazz, warm vocal, piano, upright bass" \
  --lyrics-file lyrics/song.txt \
  --seed 42

# From inline lyrics
music-gen generate \
  --style "Cyber metal, aggressive vocals" \
  --lyrics "[Verse 1]\nSteel and silicon collide\n\n[Chorus]\nWe are the machine" \
  --seed 123
```
### 5. Download Results

**runpodctl** (recommended — works on any OS, no SCP needed):

Install on your local machine:
```bash
# macOS / Linux
curl -sL https://raw.githubusercontent.com/runpod/runpodctl/main/install.sh | sudo bash

# Or download manually (macOS ARM):
cd /tmp && curl -sLO https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-darwin-all.tar.gz
tar xzf runpodctl-darwin-all.tar.gz
mkdir -p ~/bin && mv runpodctl ~/bin/
echo 'export PATH="$HOME/bin:$PATH"' >> ~/.zshrc && source ~/.zshrc
```

Transfer files:
```bash
# On the pod:
runpodctl send /app/output/song.flac
# → prints a receive code

# On your local machine:
runpodctl receive <code>
```

Other options:
- **SCP:** `scp -P <port> root@<pod-ip>:/app/output/song.flac ./` (SSH server is enabled in the Docker image)
- **Jupyter Lab:** In RunPod UI, click **Connect** → **Start Jupyter Lab**, navigate to `/app/output/`, right-click → Download
- **Python HTTP server:** On the pod: `cd /app/output && python3 -m http.server 8080`, then access via `https://<pod-id>-8080.proxy.runpod.net/`

### 6. Stop / Terminate

- **Stop** preserves disk (you can resume later) — you pay for storage only
- **Terminate** deletes everything — no further charges

### Cost Estimate

| GPU | Speed | Cost/hr | ~Cost per song |
|-----|-------|---------|----------------|
| RTX 4090 | ~71s gen | $0.44 | ~$0.01 |
| A100 40GB | ~55s gen | $1.14 | ~$0.02 |
| A100 80GB | ~55s gen | $1.64 | ~$0.03 |

*Generation time from YuE2 benchmarks. First run adds ~2-3 min for model loading.*

## Docker

Build and run with Docker (requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)):
The Dockerfile is based on `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (CUDA 12.8 + torch 2.8.0) and upgrades PyTorch to 2.10.0 with CUDA 12.8 wheels. This upgrade is required by `yue2_infer` and adds support for Blackwell GPUs (sm_120). The base image's CUDA 12.8 runtime is fully compatible with the cu128 PyTorch wheels.

```bash
# Build the image (~26 GB, without model weights)
docker build -t ghcr.io/andreisminsk/music-gen:0.1.0-torch2.10.0-cu128 .

# Generate a song (models download on first run, ~7 GB)
docker run --gpus all -v $(pwd)/output:/app/output ghcr.io/andreisminsk/music-gen:0.1.0-torch2.10.0-cu128 generate \
  --style "Jazz, warm vocal, piano, upright bass" \
  --lyrics "[Verse 1]\nWalking down the avenue\n\n[Chorus]\nTonight we break the chain" \
  --seed 42

# Generate from a lyrics file
docker run --gpus all -v $(pwd)/output:/app/output -v $(pwd)/lyrics:/app/lyrics ghcr.io/andreisminsk/music-gen:0.1.0-torch2.10.0-cu128 generate \
  --style "Jazz, warm vocal, piano" \
  --lyrics-file /app/lyrics/song.txt \
  --seed 42

# Generate lyrics then compose in one command
docker run --gpus all -v $(pwd)/output:/app/output ghcr.io/andreisminsk/music-gen:0.1.0-torch2.10.0-cu128 generate \
  --style "Indie folk rock, warm acoustic guitar, reflective male vocal" \
  --generate-lyrics --topic "rainy night in the city" --seed 42

# Shell into the container
docker run --gpus all -it ghcr.io/andreisminsk/music-gen:0.1.0-torch2.10.0-cu128 bash
```
To bake model weights into the image (avoids first-run download, increases image to ~22 GB), uncomment the `RUN` line in the Dockerfile.

### Docker Compose (with Ollama)

A `docker-compose.yml` is included to run music-gen alongside Ollama for lyrics generation:

```bash
# Start Ollama and pull a lyrics model
docker compose up -d ollama
docker compose exec ollama ollama pull gemma4:31b-cloud

# Generate lyrics + music
docker compose run music-gen lyrics-gen --style "Jazz" --language English
docker compose run music-gen generate --style "Jazz, warm vocal" --lyrics-file /app/lyrics/song.txt
```

### RunPod

To run the Docker image on [RunPod](https://runpod.io):

1. **Push the image to a registry:**
   ```bash
   docker push ghcr.io/andreisminsk/music-gen:0.1.0-torch2.10.0-cu128
   ```

2. **Deploy on RunPod:**
   - Go to **Pods** → **Deploy**
   - Select GPU: **A100 40GB** or better (24GB VRAM minimum)
   - Set **Container Disk** to **50GB+** (models need space)
   - Click **Custom Image** and enter: `ghcr.io/andreisminsk/music-gen:0.1.0-torch2.10.0-cu128`
   - Under **Environment Variables**, add:
     - `HF_TOKEN` — your [HuggingFace token](https://huggingface.co/settings/tokens) for faster downloads
   - Under **Volumes**, add a **Network Volume** mounted at `/root/.cache/huggingface` to cache models across pod restarts

3. **Run:**
   ```bash
   music-gen generate --style "Jazz, warm vocal, piano" --lyrics "..." --seed 42
   ```

> **Tip:** If you get shared memory errors, add `--shm-size=8g` to Docker run or set it in RunPod's container options.

> **Note:** The existing `deploy_runpod.sh` script installs everything from scratch on a bare PyTorch pod. Using the Docker image replaces that entire setup — just select it as the custom image and run.
  hf-cache:
  ollama-data:
```

```bash
# Start Ollama and pull a lyrics model
docker compose up -d ollama
docker compose exec ollama ollama pull gemma4:31b-cloud

# Run Ollama lyrics model and authenticate with your Ollama cloud account if require
docker compose exec ollama ollama run gemma4:31b-cloud
```

```bash
# Generate lyrics + music
docker compose run music-gen lyrics-gen --style "Jazz" --language English
docker compose run music-gen generate --style "Jazz, warm vocal" --lyrics-file /app/lyrics/song.txt
```

## 🔧 Troubleshooting

### `huggingface-hub` version conflict

`yue2_infer` pins `huggingface-hub==0.36.2`, but pip may upgrade it to 1.x which breaks `transformers`. Fix:

```bash
pip install "huggingface-hub>=0.36,<1.0"
```

The deploy script and `pipeline.py` handle this automatically.

### `torch` version conflict

`yue2_infer` requires `torch==2.10.0`. If you accidentally installed a different version:

```bash
pip install torch==2.10.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### `music-gen: command not found`

Activate your environment first:

```bash
conda activate yue2
# OR: source /workspace/music-gen/.venv/bin/activate
```

### `ModuleNotFoundError: No module named 'ollama'`

```bash
pip install ollama
```

### Ollama error `Unauthorized (status code: 401)`

Example:

```bash
 root@38f363588c13:/app# music-gen generate --style "Rock ballad" --generate-lyrics --language English --topic "Night flight to a city of heavy rains, located on a sea shore, sparkling through the clouds and rain with neon lights"
✍️  Generating lyrics (language: English)...
2026-09-20 05:43:52,557 [INFO] httpx: HTTP Request: POST http://127.0.0.1:11434/api/chat "HTTP/1.1 401 Unauthorized"
Traceback (most recent call last):
  File "/usr/local/bin/music-gen", line 7, in <module>
    sys.exit(main())
             ^^^^^^
  File "/app/src/music_gen/cli.py", line 102, in main
    lyrics = generate_lyrics(
             ^^^^^^^^^^^^^^^^
  File "/app/src/music_gen/lyrics_gen.py", line 84, in generate_lyrics
    response = ollama.chat(
               ^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/ollama/_client.py", line 387, in chat
    return self._request(
           ^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/ollama/_client.py", line 199, in _request
    return cls(**self._request_raw(*args, **kwargs).json())
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/ollama/_client.py", line 143, in _request_raw
    raise ResponseError(e.response.text, e.response.status_code) from None
ollama._types.ResponseError: Unauthorized (status code: 401)
```

This error happens when you have no Ollama lyrics model installed and Ollama missing cloud authentication.
What you need is pull the lyrics model and authenticate with Ollama.

```bash
root@38f363588c13:/app# ollama list
NAME    ID    SIZE    MODIFIED
root@38f363588c13:/app# ollama pull gemma4:31b-cloud
pulling manifest
verifying sha256 digest
writing manifest
success

root@38f363588c13:/app# ollama run gemma4:31b-cloud
You need to be signed in to Ollama to run Cloud models.

If your browser did not open, navigate to:
    https://ollama.com/connect?name=38f363588c13&key=<long_authentication_key>
```

## License

YuE2-3B weights: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
This app code: MIT
