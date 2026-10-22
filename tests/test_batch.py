"""Tests for music_gen.batch — validation, zip handling, and batch execution.

All tests use fake pipelines: no GPU, no model downloads.
"""

import json
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from music_gen.batch import BatchValidationError, load_batch, run_batch  # noqa: E402

VALID_JOB = {
    "output_file": "a.flac",
    "style": "Jazz",
    "lyrics": "[Verse 1]\nla la",
    "seed": 42,
}


def make_batch(tmp_path, jobs, defaults=None, files=None):
    """Create a batch directory with job.json and optional extra files."""
    root = tmp_path / "batch"
    root.mkdir(exist_ok=True)
    for rel, content in (files or {}).items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    data = {"version": 1, "jobs": jobs}
    if defaults is not None:
        data["defaults"] = defaults
    (root / "job.json").write_text(json.dumps(data), encoding="utf-8")
    return root


def zip_dir(src: Path, dest: Path, prefix: str = "") -> Path:
    with zipfile.ZipFile(dest, "w") as zf:
        for p in sorted(src.rglob("*")):
            if p.is_file():
                zf.write(p, f"{prefix}{p.relative_to(src)}")
    return dest


class FakeSong:
    def save(self, path):
        Path(path).write_bytes(b"FLACDATA")

    def save_artifacts(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)


class FakePipe:
    """Pipeline double; fails the calls whose index is in fail_on."""

    def __init__(self, fail_on=()):
        self.fail_on = set(fail_on)
        self.calls = 0

    def __call__(self, **kwargs):
        idx = self.calls
        self.calls += 1
        if idx in self.fail_on:
            raise RuntimeError("CUDA out of memory")
        return FakeSong()


class CapturePipe(FakePipe):
    def __init__(self):
        super().__init__()
        self.kwargs = []

    def __call__(self, **kwargs):
        self.kwargs.append(kwargs)
        return FakeSong()


class PlanPipe(FakePipe):
    def plan(self, **kwargs):
        class Plan:
            def save(self, path):
                Path(path).mkdir(parents=True, exist_ok=True)
                (Path(path) / "score.abc").write_text("X:1", encoding="utf-8")

        return Plan()


# --- loading & validation ---


def test_load_dir_merges_defaults(tmp_path):
    root = make_batch(
        tmp_path,
        jobs=[dict(VALID_JOB), {"output_file": "b.flac", "style": "Metal", "lyrics": "x"}],
        defaults={"cot": "melody", "cfg_scale": 1.5},
    )
    spec = load_batch(str(root))
    assert len(spec.jobs) == 2
    assert spec.defaults["cot"] == "melody"
    assert spec.defaults["cfg_scale"] == 1.5
    assert spec.defaults["model"] == "m-a-p/YuE2-3B"  # builtin default preserved


def test_load_zip_and_single_top_folder(tmp_path):
    root = make_batch(tmp_path, jobs=[dict(VALID_JOB)])

    spec = load_batch(str(zip_dir(root, tmp_path / "batch.zip")))
    assert len(spec.jobs) == 1
    spec.cleanup()

    wrapped = zip_dir(root, tmp_path / "wrapped.zip", prefix="wrapped/")
    spec2 = load_batch(str(wrapped))
    assert len(spec2.jobs) == 1
    spec2.cleanup()


def test_validation_collects_all_errors(tmp_path):
    root = make_batch(tmp_path, jobs=[
        {"output_file": "a.flac", "style": "X"},                       # no lyrics
        {"output_file": "a.flac", "style": "X", "lyrics": "y"},        # duplicate output_file
        {"output_file": "../evil.flac", "style": "X", "lyrics": "y"},  # bad output_file
        {"output_file": "ok.flac", "style": "X", "lyrics": "y", "model": "x"},  # defaults-only
        {"output_file": "ok2.flac", "style": "X", "lyrics": "y", "lyrics_file": "missing.txt"},
        {"output_file": "ok3.flac", "lyrics": "y"},                    # no style
    ])
    with pytest.raises(BatchValidationError) as e:
        load_batch(str(root))
    assert len(e.value.errors) == 6


def test_zip_slip_and_absolute_paths_rejected(tmp_path):
    payload = json.dumps({"version": 1, "jobs": [dict(VALID_JOB)]})
    for name, entry in (("../evil.txt", "../evil.txt"), ("/abs.txt", "/abs.txt")):
        z = tmp_path / f"{name.replace('/', '_')}.zip"
        with zipfile.ZipFile(z, "w") as zf:
            zf.writestr(entry, "x")
            zf.writestr("job.json", payload)
        with pytest.raises(BatchValidationError):
            load_batch(str(z))


# --- execution ---


def test_run_batch_success_writes_outputs_and_report(tmp_path):
    root = make_batch(
        tmp_path,
        jobs=[
            {"output_file": "a.flac", "style": "Jazz", "lyrics_file": "l.txt", "seed": 42},
            {"output_file": "b.flac", "style": "Metal", "lyrics": "raw lyrics"},
        ],
        files={"l.txt": "[Verse 1]\nfrom file"},
    )
    spec = load_batch(str(root))
    out = tmp_path / "out"
    report = run_batch(spec, FakePipe(), output_dir=str(out))

    assert report["summary"] == {"total": 2, "succeeded": 2, "failed": 0}
    assert (out / "a.flac").is_file()
    assert (out / "b.flac").is_file()
    assert (out / "batch_report.json").is_file()
    assert report["jobs"][0]["output"].endswith("a.flac")


def test_lyrics_file_wins_and_seed_autoassigned(tmp_path):
    root = make_batch(
        tmp_path,
        jobs=[
            {"output_file": "a.flac", "style": "Jazz", "lyrics_file": "l.txt",
             "prompt_json": "p.json", "seed": 7},
            {"output_file": "b.flac", "prompt_json": "p.json"},
        ],
        files={
            "l.txt": "lyrics from file",
            "p.json": json.dumps({"style": "Style from json", "lyrics": "lyrics from json"}),
        },
    )
    spec = load_batch(str(root))
    pipe = CapturePipe()
    report = run_batch(spec, pipe, output_dir=str(tmp_path / "out"))

    assert pipe.kwargs[0]["lyrics"] == "lyrics from file"  # lyrics_file > prompt_json
    assert pipe.kwargs[0]["seed"] == 7
    assert pipe.kwargs[1]["style"] == "Style from json"  # prompt_json fallback
    assert pipe.kwargs[1]["lyrics"] == "lyrics from json"
    assert isinstance(report["jobs"][1]["seed"], int)  # auto-assigned, recorded


def test_failure_isolation(tmp_path):
    root = make_batch(
        tmp_path,
        jobs=[dict(VALID_JOB), {"output_file": "b.flac", "style": "M", "lyrics": "x"}],
    )
    spec = load_batch(str(root))
    report = run_batch(spec, FakePipe(fail_on={0}), output_dir=str(tmp_path / "out"))

    assert report["summary"] == {"total": 2, "succeeded": 1, "failed": 1}
    assert report["jobs"][0]["status"] == "error"
    assert "out of memory" in report["jobs"][0]["error"]
    assert report["jobs"][1]["status"] == "ok"


def test_stop_on_error(tmp_path):
    root = make_batch(
        tmp_path,
        jobs=[dict(VALID_JOB), {"output_file": "b.flac", "style": "M", "lyrics": "x"}],
    )
    spec = load_batch(str(root))
    report = run_batch(
        spec, FakePipe(fail_on={0}),
        output_dir=str(tmp_path / "out"), stop_on_error=True,
    )
    assert len(report["jobs"]) == 1
    assert report["summary"]["failed"] == 1


def test_consecutive_oom_aborts_batch(tmp_path, monkeypatch):
    import music_gen.batch as batch_mod

    monkeypatch.setattr(batch_mod, "MAX_CONSECUTIVE_OOM", 2)
    jobs = [{"output_file": f"j{i}.flac", "style": "M", "lyrics": "x"} for i in range(4)]
    spec = load_batch(str(make_batch(tmp_path, jobs)))
    report = run_batch(
        spec, FakePipe(fail_on={0, 1, 2, 3}), output_dir=str(tmp_path / "out"))

    assert len(report["jobs"]) == 2  # aborted after 2 consecutive OOMs
    assert report["summary"]["failed"] == 2


def test_report_written_on_interrupt(tmp_path):
    class CrashPipe:
        def __call__(self, **kwargs):
            raise KeyboardInterrupt

    spec = load_batch(str(make_batch(tmp_path, jobs=[dict(VALID_JOB)])))
    out = tmp_path / "out"
    with pytest.raises(KeyboardInterrupt):
        run_batch(spec, CrashPipe(), output_dir=str(out))
    assert (out / "batch_report.json").is_file()


def test_melody_only_uses_plan(tmp_path):
    root = make_batch(
        tmp_path,
        jobs=[{"output_file": "a.flac", "style": "Jazz", "lyrics": "x", "melody_only": True}],
    )
    spec = load_batch(str(root))
    out = tmp_path / "out"
    report = run_batch(spec, PlanPipe(), output_dir=str(out))

    assert report["summary"]["succeeded"] == 1
    assert (out / "a" / "score.abc").is_file()


# --- CLI wiring ---


def test_cli_validation_exit_code(tmp_path):
    from music_gen.cli import main

    root = make_batch(tmp_path, jobs=[{"style": "X"}])  # no output_file, no lyrics
    assert main(["batch", "--dir", str(root)]) == 2


def test_cli_help(capsys):
    from music_gen.cli import main

    with pytest.raises(SystemExit) as e:
        main(["batch", "--help"])
    assert e.value.code == 0
    assert "--stop-on-error" in capsys.readouterr().out
