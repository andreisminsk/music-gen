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
music-gen plan \
  --style "Jazz-funk, warm vocal" \
  --lyrics "[Verse 1]\nSome lyrics here" \
  --seed 42
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

- `m-a-p/YuE2-Vae` — better perceptual quality (default)
- `m-a-p/YuE2-Vae-legacy` — higher benchmark scores

```bash
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

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `gemma4:31b-cloud` | Ollama model for lyrics generation |
| `--language` | `English` | Language to write in |
| `--topic` | — | Optional topic or story |
| `--sections` | `4` | Approximate number of sections |
| `--temperature` | `0.8` | Sampling temperature |
| `--output` | `lyrics/song.txt` | Output file path |
| `--print-only` | — | Print to stdout, don't save |

### End-to-end: generate lyrics → compose music

```bash
# On RunPod: generate lyrics then compose
python -m music_gen.runpod_generate \
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

### Built-in examples

```python
from music_gen.examples import get_example, list_examples

print(list_examples())  # ['jazz_funk', 'cyber_metal', 'mandarin_pop', 'russian_rock']

prompt = get_example("russian_rock")
# prompt = {"style": "...", "lyrics": "...", "seed": 2026}
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
conda activate yue2
# OR: source /workspace/music-gen/.venv/bin/activate

# Built-in example (Russian rock!)
python -m music_gen.runpod_generate --example russian_rock

# Custom prompt
python -m music_gen.runpod_generate \
  --style "Jazz, warm vocal, piano, upright bass" \
  --lyrics "[Verse 1]
Walking down the avenue

[Chorus]
Tonight we break the chain" \
  --seed 42

# CLI tool
music-gen generate \
  --style "Cyber metal, aggressive vocals" \
  --lyrics "[Verse 1]
Steel and silicon collide

[Chorus]
We are the machine" \
  --seed 123
```

### 5. Download Results

- **Jupyter Lab:** Navigate to `output/` and download FLAC files
- **SCP:** `scp root@<pod-ip>:/workspace/music-gen/output/song.flac ./`
- **RunPod CLI:** Use `runpodctl send song.flac`

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

## License

YuE2-3B weights: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
This app code: MIT
