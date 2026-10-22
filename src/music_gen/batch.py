"""Batch processing: many generation jobs, one pipeline load.

Input is a zip archive (or a directory) containing ``job.json`` — a versioned
list of ``music-gen generate`` parameter sets — plus any referenced input
files (lyrics text, ABC scores, prompt JSONs). The YuE2 pipeline is loaded
once and reused across all jobs.

Design: dev-docs/BATCH-ARCH.md
"""

from __future__ import annotations

import json
import logging
import random
import re
import shutil
import tempfile
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

BATCH_VERSION = 1
JOB_JSON = "job.json"
MAX_UNCOMPRESSED = 500 * 1024 * 1024  # 500 MB total extracted size
MAX_ENTRIES = 1000
MAX_CONSECUTIVE_OOM = 3

OUTPUT_FILE_RE = re.compile(r"^[A-Za-z0-9_-]+\.flac$")
COT_MODES = ("full", "melody", "off")

DEFAULT_FIELDS = {"cot", "cfg_scale", "output_dir", "model", "vae", "device"}
DEFAULTS_ONLY_FIELDS = {"model", "vae", "device", "output_dir"}
JOB_FIELDS = {
    "style", "lyrics", "lyrics_file", "prompt_json", "abc", "cot", "cfg_scale",
    "seed", "output_file", "melody_only", "generate_lyrics", "topic",
    "language", "lyrics_model", "lyrics_output",
}
PATH_FIELDS = ("lyrics_file", "prompt_json", "abc")

_BUILTIN_DEFAULTS: dict[str, Any] = {
    "cot": "full",
    "cfg_scale": 1.2,
    "output_dir": "output",
    "model": "m-a-p/YuE2-3B",
    "vae": "m-a-p/YuE2-Vae",
    "device": "cuda",
}


class BatchValidationError(Exception):
    """Batch source is invalid; carries ALL collected errors, not just the first."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__(f"{len(errors)} validation error(s)")


@dataclass
class BatchSpec:
    """A validated batch: source dir (job.json + input files) and job list."""

    root: Path
    defaults: dict[str, Any]
    jobs: list[dict[str, Any]]
    source: str
    _temp_dir: Optional[tempfile.TemporaryDirectory] = None

    def cleanup(self) -> None:
        """Remove the extraction dir if the source was a zip."""
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None


def load_batch(source: str) -> BatchSpec:
    """Load and validate a batch from a zip archive or a directory.

    Raises BatchValidationError (with all errors) on any problem.
    """
    path = Path(source)
    if not path.exists():
        raise BatchValidationError([f"Batch source not found: {source}"])

    if path.is_dir():
        return _load_from_dir(path, source=str(path))

    # Anything else is treated as a zip archive
    temp = tempfile.TemporaryDirectory(prefix="music-gen-batch-")
    try:
        _extract_zip(path, Path(temp.name))
        spec = _load_from_dir(Path(temp.name), source=str(path))
    except Exception:
        temp.cleanup()
        raise
    spec._temp_dir = temp
    return spec


def _extract_zip(zip_path: Path, dest: Path) -> None:
    """Extract zip safely: size/entry caps, no path traversal, no absolute paths."""
    try:
        with zipfile.ZipFile(zip_path) as zf:
            infos = zf.infolist()
            if len(infos) > MAX_ENTRIES:
                raise BatchValidationError(
                    [f"Zip has {len(infos)} entries (max {MAX_ENTRIES})"])
            total = sum(i.file_size for i in infos)
            if total > MAX_UNCOMPRESSED:
                raise BatchValidationError(
                    [f"Zip expands to {total / (1 << 20):.0f} MB "
                     f"(max {MAX_UNCOMPRESSED // (1 << 20)} MB)"])
            dest_root = dest.resolve()
            for info in infos:
                parts = [p for p in info.filename.replace("\\", "/").split("/")
                         if p not in ("", ".")]
                if not parts:
                    continue
                if info.filename.startswith(("/", "\\")) or parts[0].endswith(":"):
                    raise BatchValidationError(
                        [f"Absolute path in zip entry: {info.filename!r}"])
                if ".." in parts:
                    raise BatchValidationError(
                        [f"Path traversal in zip entry: {info.filename!r}"])
                target = dest_root.joinpath(*parts)
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out)
    except zipfile.BadZipFile as e:
        raise BatchValidationError([f"Invalid zip archive {zip_path}: {e}"]) from e


def _find_job_json(base: Path) -> Path:
    direct = base / JOB_JSON
    if direct.is_file():
        return direct
    # Convenience: allow a zip that wraps everything in a single top-level folder
    subdirs = [d for d in base.iterdir() if d.is_dir()]
    if len(subdirs) == 1 and (subdirs[0] / JOB_JSON).is_file():
        return subdirs[0] / JOB_JSON
    raise BatchValidationError([f"{JOB_JSON} not found at the root of the batch source"])


def _load_from_dir(base: Path, source: str) -> BatchSpec:
    job_json_path = _find_job_json(base)
    root = job_json_path.parent
    errors: list[str] = []

    try:
        data = json.loads(job_json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        raise BatchValidationError([f"Cannot read {JOB_JSON}: {e}"]) from e

    if not isinstance(data, dict):
        raise BatchValidationError([f"{JOB_JSON} must contain a JSON object"])

    version = data.get("version", BATCH_VERSION)
    if version != BATCH_VERSION:
        raise BatchValidationError(
            [f"Unsupported {JOB_JSON} version: {version!r} (expected {BATCH_VERSION})"])

    defaults_raw = data.get("defaults", {})
    if not isinstance(defaults_raw, dict):
        errors.append("'defaults' must be a JSON object")
        defaults_raw = {}
    unknown = set(defaults_raw) - DEFAULT_FIELDS
    if unknown:
        errors.append(
            f"defaults: unknown fields {sorted(unknown)} (allowed: {sorted(DEFAULT_FIELDS)})")
    if "cot" in defaults_raw and defaults_raw["cot"] not in COT_MODES:
        errors.append("defaults: 'cot' must be one of " + "|".join(COT_MODES))
    cs = defaults_raw.get("cfg_scale")
    if cs is not None and (isinstance(cs, bool) or not isinstance(cs, (int, float))):
        errors.append("defaults: 'cfg_scale' must be a number")
    for sf in ("output_dir", "model", "vae", "device"):
        if sf in defaults_raw and not isinstance(defaults_raw[sf], str):
            errors.append(f"defaults: {sf!r} must be a string")

    defaults = {**_BUILTIN_DEFAULTS, **defaults_raw}

    jobs_raw = data.get("jobs")
    if not isinstance(jobs_raw, list) or not jobs_raw:
        errors.append("'jobs' must be a non-empty array")
        jobs_raw = []

    seen: dict[str, int] = {}
    valid_jobs: list[dict[str, Any]] = []
    for i, job in enumerate(jobs_raw):
        where = f"jobs[{i}]"
        if not isinstance(job, dict):
            errors.append(f"{where} must be a JSON object")
            continue

        unknown = (set(job) - JOB_FIELDS) - DEFAULTS_ONLY_FIELDS
        if unknown:
            errors.append(
                f"{where}: unknown fields {sorted(unknown)} (allowed: {sorted(JOB_FIELDS)})")
        forbidden = set(job) & DEFAULTS_ONLY_FIELDS
        if forbidden:
            errors.append(
                f"{where}: {sorted(forbidden)} can only be set in 'defaults', not per job")

        of = job.get("output_file")
        if not of or not isinstance(of, str):
            errors.append(f"{where}: 'output_file' is required, e.g. \"my-song.flac\"")
        elif not OUTPUT_FILE_RE.match(of):
            errors.append(
                f"{where}: 'output_file' must be [A-Za-z0-9_-]+.flac, got {of!r}")
        else:
            key = of.lower()
            if key in seen:
                errors.append(
                    f"{where}: duplicate output_file {of!r} (already used by jobs[{seen[key]}])")
            else:
                seen[key] = i

        for pf in PATH_FIELDS:
            v = job.get(pf)
            if v is None:
                continue
            if not isinstance(v, str) or not v:
                errors.append(f"{where}: {pf!r} must be a non-empty relative path")
                continue
            p = Path(v)
            if p.is_absolute() or ".." in p.parts:
                errors.append(
                    f"{where}: {pf!r} must be relative to the batch root, no '..' (got {v!r})")
            elif not (root / p).is_file():
                errors.append(f"{where}: {pf} file not found in batch: {v!r}")

        # Parse prompt_json here so style/lyrics fallbacks can be checked
        prompt_data: Optional[dict] = None
        pj = job.get("prompt_json")
        if isinstance(pj, str) and pj and (root / pj).is_file():
            try:
                loaded = json.loads((root / pj).read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    prompt_data = loaded
                else:
                    errors.append(f"{where}: prompt_json {pj!r} must contain a JSON object")
            except (json.JSONDecodeError, OSError) as e:
                errors.append(f"{where}: cannot read prompt_json {pj!r}: {e}")

        if not job.get("style") and not (prompt_data and prompt_data.get("style")):
            errors.append(f"{where}: 'style' is required (directly or via prompt_json)")
        has_lyrics = (
            bool(job.get("lyrics"))
            or bool(job.get("lyrics_file"))
            or bool(job.get("generate_lyrics"))
            or bool(prompt_data and prompt_data.get("lyrics"))
        )
        if not has_lyrics:
            errors.append(
                f"{where}: no lyrics source — use 'lyrics', 'lyrics_file', "
                f"'prompt_json' (with lyrics) or 'generate_lyrics'")

        if "cot" in job and job["cot"] not in COT_MODES:
            errors.append(f"{where}: 'cot' must be one of " + "|".join(COT_MODES))
        if job.get("melody_only") and job.get("cot") == "off":
            errors.append(f"{where}: melody_only requires cot 'full' or 'melody', not 'off'")
        cs = job.get("cfg_scale")
        if cs is not None and (isinstance(cs, bool) or not isinstance(cs, (int, float))):
            errors.append(f"{where}: 'cfg_scale' must be a number")
        if "seed" in job and (isinstance(job["seed"], bool) or not isinstance(job["seed"], int)):
            errors.append(f"{where}: 'seed' must be an integer")
        for bf in ("melody_only", "generate_lyrics"):
            if bf in job and not isinstance(job[bf], bool):
                errors.append(f"{where}: {bf!r} must be a boolean")
        for sf in ("style", "topic", "language", "lyrics_model", "lyrics_output"):
            if sf in job and not isinstance(job[sf], str):
                errors.append(f"{where}: {sf!r} must be a string")

        valid_jobs.append(job)

    if errors:
        raise BatchValidationError(errors)

    return BatchSpec(root=root, defaults=defaults, jobs=valid_jobs, source=source)


def _resolve_job(job: dict[str, Any], defaults: dict[str, Any], root: Path) -> dict[str, Any]:
    """Merge defaults, inline referenced files, auto-assign a seed if missing."""
    r = dict(defaults)
    r.update({k: v for k, v in job.items() if v is not None})

    if r.get("prompt_json"):
        pdata = json.loads((root / r["prompt_json"]).read_text(encoding="utf-8"))
        for key in ("style", "lyrics", "seed"):
            if r.get(key) is None:
                r[key] = pdata.get(key)

    if r.get("lyrics_file"):  # wins over prompt_json lyrics, same precedence as the CLI
        r["lyrics"] = (root / r["lyrics_file"]).read_text(encoding="utf-8")

    if r.get("abc"):
        r["abc"] = (root / r["abc"]).read_text(encoding="utf-8")

    if r.get("seed") is None:
        r["seed"] = random.randint(0, 2**31 - 1)

    return r


def run_batch(
    spec: BatchSpec,
    pipe,
    output_dir: Optional[str] = None,
    stop_on_error: bool = False,
    report_path: Optional[str] = None,
) -> dict[str, Any]:
    """Run all jobs sequentially on one shared pipeline.

    Per-job failures are isolated and recorded; the batch continues unless
    stop_on_error is set or MAX_CONSECUTIVE_OOM CUDA OOMs occur in a row.
    The report is written even if the batch is interrupted (try/finally).

    Returns:
        Report dict: {"batch", "output_dir", "started", "finished", "duration_s",
                      "summary": {total, succeeded, failed}, "jobs": [...]}
    """
    out_root = Path(output_dir) if output_dir else Path(spec.defaults["output_dir"])
    out_root.mkdir(parents=True, exist_ok=True)
    if report_path is None:
        report_path = str(out_root / "batch_report.json")

    report: dict[str, Any] = {
        "batch": spec.source,
        "output_dir": str(out_root),
        "started": _utcnow(),
        "finished": None,
        "duration_s": None,
        "summary": {"total": len(spec.jobs), "succeeded": 0, "failed": 0},
        "jobs": [],
    }

    total = len(spec.jobs)
    consecutive_oom = 0
    t_start = time.monotonic()

    try:
        for i, raw in enumerate(spec.jobs):
            job = _resolve_job(raw, spec.defaults, spec.root)
            label = job["output_file"]
            prefix = f"[{i + 1}/{total}]"
            logger.info("%s %s — starting (seed=%s, cot=%s)",
                        prefix, label, job["seed"], job["cot"])
            entry: dict[str, Any] = {
                "index": i,
                "output_file": label,
                "seed": job["seed"],
                "status": None,
            }
            t0 = time.monotonic()
            err: Optional[Exception] = None
            try:
                out = _run_job(pipe, job, out_root)
            except Exception as e:  # noqa: BLE001 — every job failure is isolated
                err = e
            entry["duration_s"] = round(time.monotonic() - t0, 1)
            report["jobs"].append(entry)

            if err is None:
                entry["status"] = "ok"
                entry["output"] = str(out)
                report["summary"]["succeeded"] += 1
                consecutive_oom = 0
                logger.info("%s %s — ok (%.1fs) → %s",
                            prefix, label, entry["duration_s"], out)
            else:
                entry["status"] = "error"
                entry["error"] = str(err)
                report["summary"]["failed"] += 1
                logger.error("%s %s — FAILED (%.1fs): %s",
                             prefix, label, entry["duration_s"], err)
                if _is_oom(err):
                    consecutive_oom += 1
                    if consecutive_oom >= MAX_CONSECUTIVE_OOM:
                        logger.error(
                            "Aborting batch: %d consecutive CUDA OOM failures — "
                            "GPU state is likely unrecoverable.", consecutive_oom)
                        break
                if stop_on_error:
                    logger.error("Stopping batch (--stop-on-error).")
                    break
            _empty_cache()
    finally:
        report["finished"] = _utcnow()
        report["duration_s"] = round(time.monotonic() - t_start, 1)
        _write_report(report, report_path)

    logger.info("Batch done: %d/%d succeeded.",
                report["summary"]["succeeded"], report["summary"]["total"])
    return report


def _run_job(pipe, job: dict[str, Any], out_root: Path) -> Path:
    """Execute one resolved job. Returns the output path (FLAC or plan dir)."""
    from .lyrics_gen import DEFAULT_MODEL, generate_lyrics, save_lyrics
    from .pipeline import generate_song

    style = job["style"]
    lyrics = job.get("lyrics")

    if job.get("generate_lyrics"):
        lyrics = generate_lyrics(
            style=style,
            topic=job.get("topic", ""),
            language=job.get("language", "English"),
            model=job.get("lyrics_model") or DEFAULT_MODEL,
        )
        # lyrics_output is CWD-relative (same as the CLI); default: next to the song
        lyrics_out = job.get("lyrics_output") or str(
            out_root / (Path(job["output_file"]).stem + ".txt"))
        save_lyrics(lyrics, lyrics_out)
        logger.info("Lyrics saved to: %s", lyrics_out)

    if not lyrics:
        raise ValueError("job produced no lyrics")

    stem = Path(job["output_file"]).stem

    if job.get("melody_only"):
        plan = pipe.plan(style=style, lyrics=lyrics, cot=job["cot"], seed=job["seed"])
        plan_dir = out_root / stem
        plan.save(str(plan_dir))
        return plan_dir

    return Path(generate_song(
        pipe,
        style=style,
        lyrics=lyrics,
        cot=job["cot"],
        seed=job["seed"],
        cfg_scale=job["cfg_scale"],
        abc=job.get("abc"),
        output_dir=str(out_root),
        filename=stem,
    ))


def _is_oom(err: BaseException) -> bool:
    return "out of memory" in str(err).lower()


def _empty_cache() -> None:
    """Best-effort CUDA cache cleanup between jobs (fragmentation insurance)."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 — cache cleanup must never fail a batch
        pass


def _write_report(report: dict[str, Any], path: str) -> None:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                     encoding="utf-8")
        logger.info("Batch report: %s", p)
    except OSError as e:
        logger.error("Failed to write batch report to %s: %s", path, e)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
