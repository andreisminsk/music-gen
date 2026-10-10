# RunPod Serverless for music-gen — Design & Implementation

> Status: proposal / implementation guide. Based on the
> [Serverless quickstart](https://docs.runpod.io/serverless/quickstart) and the
> existing pod-based deployment (`Dockerfile`, `deploy_runpod.sh`).

## 1. Why serverless (and why not)

Current deployment model is a **persistent pod**: SSH in, run `music-gen generate`,
SCP the FLAC back. You pay per hour while idle.

RunPod **Serverless** flips this: an HTTP endpoint with a queue, workers scale
from 0→N, you pay only while a worker processes a job (plus a small cold-start
window). This fits music generation well:

| Aspect | Pod (current) | Serverless |
|---|---|---|
| Idle cost | full GPU hourly rate | $0 (scaled to zero) |
| Invocation | manual SSH + CLI | HTTPS API, programmatic |
| Cold start | once per pod lifetime | model load (~2–3 min) after idle timeout |
| Concurrency | 1 (you) | N workers (configurable) |
| Output transfer | SCP / runpodctl / Jupyter | HTTP response or object storage |

**Good fit** if you call generation from an app/bot occasionally.
**Bad fit** if you generate continuously for hours — a pod is cheaper then.

## 2. Hard constraints that shape the design

1. **~20 MB response cap.** The `/runsync` gateway caps payloads at ~20 MB in
   both directions. A 3-minute stereo FLAC is 30–80 MB → **audio cannot be
   returned inline**. Options (in order of preference):
   - Upload to object storage (S3 / Cloudflare R2) and return a presigned URL.
   - Write to a RunPod **network volume** and return the path (only useful if
     the caller can read the volume, e.g. another worker).
   - Return base64 only for small outputs (ABC plan, MIDI) — these are KB-sized.
2. **24 GB VRAM minimum** (BF16). Eligible GPU pools: RTX 4090, L40S, A100 40/80GB.
3. **Model load is expensive** (~2–3 min, ~7 GB weights). The handler must load
   the pipeline **once per worker** and reuse it across jobs (global singleton).
4. **Concurrency = 1 per worker.** The YuE2 pipeline owns the GPU; concurrent
   jobs on one worker would OOM. Use `concurrency_modifier` returning 1 and let
   RunPod scale worker count instead.
5. **No Ollama sidecar in serverless.** The `lyrics-gen` path needs an LLM.
   Simplest: require lyrics in the request. Optional: call a remote Ollama
   (`OLLAMA_HOST` env, e.g. Ollama Cloud) from the handler.

## 3. Architecture

```
client ──POST /run (style, lyrics, seed)──▶ RunPod endpoint (queue)
                                              │
                                              ▼
                                     worker (this image)
                                     ├─ handler.py
                                     │   ├─ get_pipe()  ← lazy, cached per worker
                                     │   ├─ pipe.plan() / pipe(**kwargs)
                                     │   ├─ FLAC → S3/R2 (presigned URL)
                                     │   └─ return {audio_url, abc, ...}
                                              │
client ◀──{id, status, output}────────────────┘
```

Two request modes:
- **Sync** (`/runsync?wait=600`): blocks up to `wait` seconds; fine for
  `plan` (ABC only, ~10–30 s) and acceptable for full songs (~1–2 min warm).
- **Async** (`/run` + poll `/status/<id>`): recommended for full songs from
  web/mobile clients.

## 4. Files

```
serverless/
├── handler.py          # runpod handler (below)
├── Dockerfile          # serverless worker image (below)
├── requirements.txt    # runpod SDK (image deps come from pyproject)
└── test_input.json     # local test fixture
```

### 4.1 `serverless/handler.py`

```python
"""RunPod Serverless handler for music-gen (YuE2-3B).

Input (event["input"]):
    style        str, required — e.g. "Jazz, warm vocal, piano, upright bass"
    lyrics       str, required — with [Verse]/[Chorus] markers
    seed         int, optional
    cot          "full" | "melody" | "off"        (default "full")
    cfg_scale    float, optional                  (default 1.2)
    abc          str, optional — ABC score for covers/edits
    mode         "song" | "plan"                  (default "song")
    filename     str, optional                    (default "song")

Output:
    mode="song"  → {"audio_url": ..., "audio_format": "flac", "abc": ...}
                   (audio_url present only if S3/R2 configured; otherwise
                    "error" explaining the 20 MB cap)
    mode="plan"  → {"abc": "...", "midi_b64": "..."}  (small, inline)
"""

import base64
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("music_gen.serverless")

import runpod

from music_gen.pipeline import load_pipeline, generate_song

_pipe = None  # cached per worker process


def get_pipe():
    """Load the YuE2 pipeline once per worker; reuse across jobs."""
    global _pipe
    if _pipe is None:
        vae = os.environ.get("YUE2_VAE", "m-a-p/YuE2-Vae")
        logger.info("Loading YuE2 pipeline (first job on this worker)...")
        _pipe = load_pipeline(vae=vae, device="cuda", progress=False)
        logger.info("Pipeline loaded.")
    return _pipe


def _upload_audio(flac_path: str, key: str) -> str | None:
    """Upload FLAC to S3/R2 and return a presigned URL.

    Config via env vars:
        S3_ENDPOINT_URL   e.g. https://<account>.r2.cloudflarestorage.com
        S3_BUCKET         bucket name
        S3_ACCESS_KEY_ID / S3_SECRET_ACCESS_KEY
        S3_PRESIGN_EXPIRY seconds (default 86400)
    Returns None if storage is not configured.
    """
    endpoint = os.environ.get("S3_ENDPOINT_URL")
    bucket = os.environ.get("S3_BUCKET")
    if not (endpoint and bucket):
        return None

    import boto3
    from botocore.client import Config

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ["S3_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4"),
        region_name=os.environ.get("S3_REGION", "auto"),
    )
    object_key = f"music-gen/{key}.flac"
    s3.upload_file(flac_path, bucket, object_key,
                   ExtraArgs={"ContentType": "audio/flac"})
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": object_key},
        ExpiresIn=int(os.environ.get("S3_PRESIGN_EXPIRY", "86400")),
    )
    return url


def handler(event):
    job_input = event.get("input") or {}
    style = job_input.get("style")
    lyrics = job_input.get("lyrics")
    mode = job_input.get("mode", "song")

    if not style or not lyrics:
        return {"error": "Both 'style' and 'lyrics' are required in input."}

    seed = job_input.get("seed")
    cot = job_input.get("cot", "full")
    cfg_scale = float(job_input.get("cfg_scale", 1.2))
    abc = job_input.get("abc")
    filename = job_input.get("filename", f"song-{event.get('id', 'job')}")

    pipe = get_pipe()

    if mode == "plan":
        plan = pipe.plan(style=style, lyrics=lyrics, cot=cot, seed=seed)
        with tempfile.TemporaryDirectory() as td:
            plan.save(str(Path(td) / "plan"))
            plan_dir = Path(td) / "plan"
            abc_text = (plan_dir / "score.abc").read_text(encoding="utf-8") \
                if (plan_dir / "score.abc").exists() else None
            midi_b64 = None
            for p in plan_dir.glob("*.mid*"):
                midi_b64 = base64.b64encode(p.read_bytes()).decode()
                break
        return {"abc": abc_text, "midi_b64": midi_b64}

    # mode == "song"
    with tempfile.TemporaryDirectory() as td:
        flac_path = generate_song(
            pipe, style=style, lyrics=lyrics, cot=cot, seed=seed,
            cfg_scale=cfg_scale, abc=abc, output_dir=td, filename=filename,
        )
        audio_url = _upload_audio(flac_path, filename)
        # ABC artifact is small — return inline for convenience
        abc_artifact = Path(flac_path).with_suffix("")  # song.save_artifacts writes alongside
        abc_text = None
        for p in abc_artifact.parent.glob(f"{abc_artifact.name}*/*.abc") + \
                 list(abc_artifact.parent.glob("*.abc")):
            abc_text = p.read_text(encoding="utf-8")
            break

    if audio_url is None:
        return {
            "error": (
                "Audio exceeds the ~20 MB serverless response cap and no object "
                "storage is configured. Set S3_ENDPOINT_URL/S3_BUCKET (+ keys) "
                "to receive a presigned URL, or use mode='plan'."
            ),
            "abc": abc_text,
        }

    return {"audio_url": audio_url, "audio_format": "flac", "abc": abc_text}


def concurrency_modifier(current_concurrency: int) -> int:
    """YuE2 pipeline owns the whole GPU — one job per worker."""
    return 1


if __name__ == "__main__":
    runpod.serverless.start(
        {
            "handler": handler,
            "concurrency_modifier": concurrency_modifier,
            "return_aggregate_stream": True,
        }
    )
```

> Note: `generate_song()` writes artifacts next to the FLAC via
> `song.save_artifacts()`; the glob above is defensive — verify actual artifact
> layout on first run and simplify.

### 4.2 `serverless/Dockerfile`

Reuses the same base and torch pin as the pod image, minus SSH/entrypoint/Ollama:

```dockerfile
# Serverless worker image for music-gen (YuE2-3B)
# Build: docker build --platform linux/amd64 -f serverless/Dockerfile \
#            -t ghcr.io/andreisminsk/music-gen-serverless:0.2.0 .

FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /app

# torch 2.10 + cu128 (required by yue2_infer; supports Blackwell sm_120)
RUN pip install --no-cache-dir \
    torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu128

COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir -e . && \
    pip install --no-cache-dir \
    "huggingface-hub>=0.36,<1.0" "transformers>=4.57,<5.0" "hf_transfer>=0.1" \
    "runpod>=1.7" "boto3>=1.34"

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg flac \
    && rm -rf /var/lib/apt/lists/*

# Bake model weights (~7 GB) — BEFORE code COPY so this layer stays cached
# across code-only rebuilds. RunPod pulls the image to machine disk before
# the worker boots: no billed download at cold start, only the VRAM load.
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('m-a-p/YuE2-3B'); snapshot_download('m-a-p/YuE2-Vae')"

COPY serverless/handler.py /handler.py

ENV HF_HUB_ENABLE_HF_TRANSFER=1

CMD ["python3", "-u", "/handler.py"]
```

### 4.3 `serverless/test_input.json`

```json
{
    "input": {
        "style": "Jazz, warm vocal, piano, upright bass",
        "lyrics": "[Verse 1]\nWalking down the avenue\n\n[Chorus]\nTonight we break the chain",
        "seed": 42,
        "mode": "plan"
    }
}
```

(`mode: "plan"` for fast local tests — no GPU needed for the handler logic
check; switch to `"song"` on the deployed endpoint.)

## 5. Model weights strategy

**Decision: bake weights into the image — no network volume.**

| Option | Fixed cost | Cold start | Verdict |
|---|---|---|---|
| **Bake into image** ✅ | $0 | ~2–3 min VRAM load only | RunPod pulls the image to machine disk *before* the worker boots — no billed download. Order the weights layer before code COPY so rebuilds stay cached. |
| Network volume | $1.50–3.50/mo | ~2–3 min load | Rejected: our pattern (bursts of 5–7 songs, then days/weeks idle) makes the fixed fee exceed the download cost it would save (~$0.30–0.80/mo). Revisit for frequent sessions or multiple endpoints sharing weights. |
| Download on first job | $0 | +7 GB download (~5–10 min) | Fallback if image size becomes painful; requires execution timeout ≥ 1200 s. |

With baked weights, set `HF_HUB_OFFLINE=1` in the endpoint env — skips HF etag
revalidation during `from_pretrained`, shaving seconds off every cold start.

## 6. Local testing

```bash
pip install runpod
cd serverless
python handler.py        # auto-loads test_input.json, runs one fake job
```

For a GPU test with the real model, run the same handler on a pod:

```bash
python handler.py        # on a GPU pod, with test_input.json mode="song"
```

## 7. Deploy

1. Build & push:
   ```bash
   docker build --platform linux/amd64 -f serverless/Dockerfile \
     -t ghcr.io/andreisminsk/music-gen-serverless:0.2.0 .
   docker push ghcr.io/andreisminsk/music-gen-serverless:0.2.0
   ```
2. Console → **Serverless** → **New Endpoint** → **Import from Docker Registry**
   → `ghcr.io/andreisminsk/music-gen-serverless:0.2.0`
3. Endpoint configuration:
   - **Endpoint type:** Queue
   - **GPU:** RTX 4090 (24 GB, cheapest) or A100 40GB/80GB; L40S also qualifies
   - **Workers:** min 0, **max 1** — one model load per burst; jobs queue serially (N parallel workers would each pay their own cold start)
   - **Idle timeout:** 10 min (covers the "listen, regenerate one" tail of a session)
   - **Execution timeout:** ≥ 900 s (first job = model load + generation; ≥ 1200 s if not baking weights)
   - **Container disk:** 20 GB
   - **Env vars:** optional `S3_*` (see §4.1), `YUE2_VAE`, `HF_HUB_OFFLINE=1` (weights baked)
4. Test from the **Requests** tab, then programmatically:

```python
import requests, time, json

ENDPOINT = "https://api.runpod.ai/v2/<endpoint-id>"
HDRS = {"Authorization": "Bearer <api-key>"}

payload = {"input": {
    "style": "Russian rock, powerful male vocal",
    "lyrics": "[Verse 1]\nСтальной рассвет над городом\n\n[Chorus]\nМы машины",
    "seed": 2026, "mode": "song",
}}

job = requests.post(f"{ENDPOINT}/run", headers=HDRS, json=payload, timeout=30).json()
while True:
    st = requests.get(f"{ENDPOINT}/status/{job['id']}", headers=HDRS, timeout=30).json()
    if st["status"] in ("COMPLETED", "FAILED"):
        break
    time.sleep(5)
print(json.dumps(st, indent=2, ensure_ascii=False))
# → {"status": "COMPLETED", "output": {"audio_url": "https://...presigned...", ...}}
```

## 8. Lyrics generation in serverless

The Ollama-based `lyrics-gen` doesn't fit a serverless worker (no sidecar,
model download per cold start). Options:

1. **Recommended:** generate lyrics client-side (or via a separate cheap
   serverless endpoint running a small LLM), pass final lyrics in the request.
2. **Remote Ollama:** set `OLLAMA_HOST` to an Ollama Cloud / self-hosted URL;
   the `ollama` Python client accepts a host. Add to the handler:
   ```python
   if job_input.get("generate_lyrics"):
       import ollama
       client = ollama.Client(host=os.environ["OLLAMA_HOST"])
       lyrics = client.chat(model=os.environ.get("LYRICS_MODEL", "gemma4:31b-cloud"),
                            messages=[...])["message"]["content"]
   ```
   Requires auth handling for cloud models (see README troubleshooting §401).

## 9. Cost model

Per-song (warm worker, A100 40GB): ~55 s generation → ~$0.02.
Cold start adds ~2–3 min model load once per idle period (~$0.04–0.06).
Bursts amortize the cold start across N songs. Compare: pod at $1.14/hr
breaks even against ~50 warm serverless songs/hour — serverless wins for
sporadic use, pods win for sustained batch generation.

## 10. Open items

- [ ] Verify `plan.save()` artifact layout (ABC/MIDI filenames) on first run
- [ ] Decide R2 vs S3 for audio delivery (network volume ruled out — see §11)
- [ ] Consider `runpod.serverless.progress` updates for long generations
- [ ] Flashboot (faster cold boot) — test compatibility with the torch 2.10 image
- [ ] Health check: `runpod.serverless.start` supports a `test` input / worker
      warmup hook — add a `mode="warmup"` branch that only calls `get_pipe()`

## 11. Decisions (design interview, 2026-10-11)

1. **Separate Dockerfiles** for pod and serverless images; heavy layers (torch, deps) kept byte-identical so the build cache dedupes. Multi-stage single file rejected for simplicity.
2. **Bake model weights into the serverless image** — no network volume. Bursts of 5–7 songs followed by days/weeks idle make the volume's fixed monthly fee uneconomical (§5).
3. **Max workers = 1, min = 0, idle timeout 10 min** — one cold start per session (~$0.15–0.20/session), $0 between sessions.
4. **No Ollama sidecar in the worker** — lyrics come from the client or a remote `OLLAMA_HOST`.
5. **Pod + CLI/SSH/SCP evaluated as an alternative** (§12) — preferred for interactive
   sessions; serverless remains the choice for unattended API-driven generation.

## 12. Alternative: Pod + CLI/SSH/SCP (no 20 MB cap)

Evaluated against the serverless design. The ~20 MB response cap only hurts
*API-driven* audio delivery; for interactive sessions the classic pod workflow
avoids it entirely — SCP has no size limit, and the existing pod image
(`Dockerfile` at repo root) already ships SSH, key injection, and SCP support.

### 12.1 Workflow

```bash
runpodctl config --apiKey RUNPOD_API_KEY     # once

runpodctl pod start POD_ID                   # or: runpodctl pod create --name music-gen \
                                             #       --gpu-id "RTX 4090" --image ghcr.io/andreisminsk/music-gen:0.2.0-torch2.10.0-cu128
# wait ~1–3 min for the pod to reach Running and SSH to accept connections
ssh root@<pod-gateway> -p <port> "music-gen generate --style 'Jazz' --lyrics-file lyrics/song.txt --seed 42"
scp -P <port> root@<pod-gateway>:/app/output/*.flac ./
runpodctl pod stop POD_ID                    # GPU released; storage-only billing
```

Full lifecycle via CLI (verified against docs.runpod.io/pods/manage-pods):

| Action | CLI |
|---|---|
| Deploy | `runpodctl pod create --name NAME --gpu-id "GPU" --image "IMAGE"` |
| Start | `runpodctl pod start POD_ID` |
| Stop | `runpodctl pod stop POD_ID` |
| Terminate | `runpodctl pod delete POD_ID` |
| List | `runpodctl pod list` |

The Python SDK (`runpod` package) exposes the same operations
(`runpod.start_pod`, `runpod.stop_pod`, `runpod.terminate_pod`, …) for
programmatic control.

### 12.2 Comparison vs serverless

| Aspect | Serverless | Pod + CLI/SSH/SCP |
|---|---|---|
| 20 MB cap | yes — needs S3/R2 for audio | **none — SCP any size** |
| New infra | handler + Dockerfile + S3 account | **zero — existing image works as-is** |
| Session cost (7 songs) | ~$0.15–0.20 | ~$0.10–0.20 (start 2–3 min + 7 gens + transfer) |
| Idle cost | $0 guaranteed (scale to zero) | $0 **only if stopped** — a forgotten running 4090 = ~$10.56/day |
| Stop semantics | n/a | container disk cleared → weights must be **baked in image** (re-pulled on start) or on a network volume ($1.50–3.50/mo) |
| Programmatic API | native (queue, status polling) | possible via SDK, but clunky |

### 12.3 Verdict

- **Interactive use** (generate a burst, listen, pick, forget for weeks):
  **pod workflow wins** — no S3, no handler, no new image; SCP is natural.
- **Unattended API-driven generation** (bot/app integration): **serverless wins** —
  native queue, scale-to-zero without discipline, status polling.

The one real pod risk is a forgotten running pod; mitigate by making
`runpodctl pod stop` part of the session script (a `finally` block).

> Open question: bake weights into the pod image too (image grows ~7 GB, but
> `pod start` then skips the model download — same reasoning as §5), or keep
> the current slim image and accept a one-time download per start.
