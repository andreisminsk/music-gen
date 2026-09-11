"""YuE2 pipeline wrapper with lazy loading and device management."""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Reconfigure stdout for Unicode output (emoji, non-ASCII lyrics)
sys.stdout.reconfigure(encoding="utf-8")

MODEL_ID = "m-a-p/YuE2-3B"
VAE_ID = "m-a-p/YuE2-Vae"
VAE_LEGACY_ID = "m-a-p/YuE2-Vae-legacy"


def _ensure_yue2_infer() -> None:
    """Download and install the yue2_infer wheel if not already available."""
    try:
        import yue2  # noqa: F401
        return
    except ImportError:
        pass

    import subprocess
    from huggingface_hub import hf_hub_download

    logger.info("Downloading yue2_infer wheel from HuggingFace...")
    wheel_path = hf_hub_download(MODEL_ID, "yue2_infer-0.1.5-py3-none-any.whl")
    logger.info("Installing yue2_infer...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", wheel_path],
        stdout=subprocess.DEVNULL,
    )
    # yue2_infer may downgrade huggingface-hub or change torch; restore compatible versions
    logger.info("Restoring dependency compatibility...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet",
         "huggingface-hub>=0.36,<1.0",
         "torch==2.10.0"],
        stdout=subprocess.DEVNULL,
    )
    logger.info("yue2_infer installed successfully.")


def load_pipeline(
    model_id: str = MODEL_ID,
    vae: str = VAE_ID,
    device: str = "cuda",
    progress: bool = True,
):
    """Load the YuE2 pipeline, installing yue2_infer if needed.

    Args:
        model_id: HuggingFace model repo ID.
        vae: VAE variant to use ("m-a-p/YuE2-Vae" or "m-a-p/YuE2-Vae-legacy").
        device: Torch device string.
        progress: Show generation progress messages.

    Returns:
        YuE2Pipeline instance.
    """
    _ensure_yue2_infer()
    from yue2 import YuE2Pipeline

    logger.info(f"Loading YuE2 pipeline: {model_id} with VAE: {vae}")
    pipe = YuE2Pipeline.from_pretrained(model_id, vae=vae, device=device, progress=progress)
    logger.info("Pipeline loaded.")
    return pipe


def generate_song(
    pipe,
    style: str,
    lyrics: str,
    cot: str = "full",
    seed: Optional[int] = None,
    cfg_scale: float = 1.2,
    abc: Optional[str] = None,
    output_dir: str = "output",
    filename: str = "song",
):
    """Generate a song using the YuE2 pipeline.

    Args:
        pipe: Loaded YuE2Pipeline instance.
        style: Style prompt (e.g. "Jazz, warm vocal, piano, upright bass").
        lyrics: Lyrics text with section markers like [Verse], [Chorus].
        cot: Chain-of-thought mode: "full" (melody+chords), "melody", or "off".
        seed: Random seed for reproducibility.
        cfg_scale: Classifier-free guidance scale (default 1.2).
        abc: Optional ABC notation score for covers/edits.
        output_dir: Directory to save output files.
        filename: Base filename (without extension).

    Returns:
        Path to the saved FLAC file.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    kwargs = dict(style=style, lyrics=lyrics, cot=cot, cfg_scale=cfg_scale)
    if seed is not None:
        kwargs["seed"] = seed
    if abc is not None:
        kwargs["abc"] = abc

    logger.info(f"Generating song (cot={cot}, seed={seed}, cfg_scale={cfg_scale})...")
    song = pipe(**kwargs)

    flac_path = out_path / f"{filename}.flac"
    song.save(str(flac_path))
    song.save_artifacts(str(out_path / filename))
    logger.info(f"Song saved to {flac_path}")

    return str(flac_path)


def load_prompt_from_json(json_path: str) -> dict:
    """Load a prompt JSON file with style, lyrics, and seed."""
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    return data
