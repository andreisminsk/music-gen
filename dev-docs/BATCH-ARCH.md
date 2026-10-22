# Batch Processing for music-gen — Architecture

> Status: design. Companion to `SERVERLESS-RUNPOD.md` (the batch runner is
> serverless-ready: `run_batch()` takes an extracted dir and returns a report).

## 1. Problem framing

Today each `music-gen generate` invocation pays the full **model load tax**
(~2–3 min, ~7 GB weights) before ~1 min of actual generation. A 10-song
session = 10 model loads ≈ 25 min of pure overhead. Additionally, multi-file
workflows (lyrics files, ABC scores, prompt JSONs) are awkward to transfer to
a RunPod pod — one zip upload is simpler than many SCPs.

**Requirements:**
- Single call processes N generation requests
- Pipeline loaded **once**, reused across jobs
- Input = zip with `job.json` (array of generate-parameter sets) + referenced files
- One failed job must not kill the batch
- Results must be inspectable programmatically

**Non-goals (v1):** parallel GPU execution (pipeline owns the GPU, concurrency
= 1), per-job model switching (reloading weights per job is the cost we're
eliminating).

## 2. Component breakdown

```
batch.zip ──extract──▶ temp dir
                         │
                      job.json ──validate──▶ JobSpec list      ← FAIL FAST (before model load)
                         │
                         ▼
              music-gen batch ──▶ load_pipeline()  ← ONCE
                                     │
                                     ▼
                               for job in jobs (array order):
                                  resolve paths (lyrics_file / abc / prompt_json)
                                  [optional] generate_lyrics via Ollama
                                  generate_song(pipe, ...) ──▶ output/<output_file> + artifacts
                                  record result ──▶ batch_report.json
                                     │
                                     ▼
                               summary + exit code
```

**New module: `src/music_gen/batch.py`**
- `load_batch(source) -> BatchSpec` — zip extraction (or directory passthrough) + schema validation
- `validate_batch(spec) -> list[str]` — collects ALL errors (not first-fail), returns before any GPU work
- `run_batch(spec, pipe) -> BatchReport` — sequential execution, per-job error isolation, report building
- `cli.py` gains a `batch` subcommand; `pipeline.py` untouched.

**CLI:**
```bash
music-gen batch --zip batch.zip -o output [--stop-on-error] [--report output/batch_report.json]
music-gen batch --dir batch/          # unzipped, for local iteration
```

## 3. job.json schema (versioned)

```json
{
  "version": 1,
  "defaults": {
    "cot": "full",
    "cfg_scale": 1.2,
    "output_dir": "output",
    "model": "m-a-p/YuE2-3B",
    "vae": "m-a-p/YuE2-Vae",
    "device": "cuda"
  },
  "jobs": [
    {
      "output_file": "tonight-awake.flac",
      "style": "Jazz, warm vocal, piano, upright bass",
      "lyrics_file": "lyrics/tonight.txt",
      "seed": 42
    },
    {
      "output_file": "machine.flac",
      "style": "Cyber metal, aggressive vocals",
      "lyrics": "…",
      "seed": 123
    }
  ]
}
```

 **Field mapping** mirrors generate flags in snake_case: `style`, `lyrics`,
 `lyrics_file`, `prompt_json`, `abc`, `cot`, `cfg_scale`, `seed`, `output_file`,
 `melody_only`, `generate_lyrics`, `topic`, `language`, `lyrics_model`,
 `lyrics_output`.

**Key decisions:**

| Decision | Rationale |
|---|---|
| `defaults` block | All jobs typically share model/vae/cot; removes 90% of repetition |
| `model`/`vae`/`device` **defaults-only** | Per-job override would trigger a model reload — silently reintroduces the cost we're eliminating. Validation **errors** if present per-job |
 | `output_dir` defaults-only, `output_file` per-job | Contained output tree; per-job output dirs invite path confusion |
 | `output_file` required, unique, name **with extension** (`.flac`), stem `[A-Za-z0-9_-]` | Explicit and self-describing — the user sees exactly what file they'll get. Duplicates are a validation error, not silent overwrite. Artifacts land in `output/<stem>/` alongside the FLAC |
| Paths relative to zip root, no `..`, no absolute | Zip-slip protection; files resolve inside extraction dir |
| Missing seed → auto-assign random int, **recorded in report** | Free reproducibility — the underlying pipeline's random choice is unobservable, so we choose it ourselves |
| `version` field | Schema evolution without breaking old zips |

## 4. Failure isolation & exit codes

 - **Validation errors** (bad schema, missing files, duplicate output files):
  collected and reported together, **before** model load. Exit **2**. This is
  the most important rule — never load 7 GB of weights for a typo.
- **Job runtime errors** (OOM, Ollama 401, bad lyrics): caught per job,
  recorded, batch continues. Exit **1** if any failed, **0** if all succeeded.
- `--stop-on-error` opts into fail-fast for expensive batches.
- Report written even on Ctrl+C (try/finally) — partial results are valuable
  when each job costs GPU minutes.

**`batch_report.json`:**
```json
{
  "batch": "batch.zip", "started": "...", "finished": "...",
  "summary": {"total": 10, "succeeded": 9, "failed": 1},
   "jobs": [
     {"index": 0, "output_file": "tonight-awake.flac", "status": "ok",
      "output": "output/tonight-awake.flac", "seed": 42, "duration_s": 61.3},
     {"index": 1, "output_file": "machine.flac", "status": "error",
      "error": "error message", "duration_s": 12.0}
   ]
}
```

## 5. Technology choices

- **stdlib `zipfile` + manual validation** — no new dependencies. The schema
  is small and flat; hand-rolled validation with collected error messages is
  ~80 lines and keeps the lean dep profile. (Pydantic is the alternative if
  the schema grows — revisit at v2.)
- **Sequential execution** — the YuE2 pipeline owns the GPU; parallelism means
  OOM, not speedup.
- **Extraction to temp dir** (or `--dir` passthrough) — uniform path resolution
  for both transport modes.
- **`torch.cuda.empty_cache()` between jobs** — cheap insurance against
  fragmentation across a long batch.

## 6. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Validation gap discovered mid-batch | Validate everything (schema + file existence + uniqueness) before `load_pipeline()` |
| CUDA OOM mid-batch poisons subsequent jobs | Catch, record, continue; if consecutive OOMs ≥ 3, abort batch (GPU state likely unrecoverable) |
| Zip bomb / huge archive | Cap extraction size (e.g. 500 MB) and file count; reject before extracting |
 | Non-ASCII / weird zip entry names | Sanitize entry names on extraction; `output_file` charset validation keeps outputs safe |
| 2-hour batch dies at job 29 | Per-job isolation + incremental report + `--stop-on-error` for cost control |

## 7. Operational concerns

- **Docker/RunPod:** zero image changes — pure Python, stdlib only. Workflow:
  upload one zip → `music-gen batch --zip batch.zip` → download `output/`.
- **Serverless-ready:** `run_batch()` is a library function taking an extracted
  dir; the future serverless handler (see `SERVERLESS-RUNPOD.md`) can accept a
  zip and call it directly. The report JSON maps naturally to a job-status
  response.
- **Observability:** per-job log lines (`[3/10] machine … ok 61s`) + the report
  file cover both human and programmatic consumption.

**Effort estimate:** ~1–1.5 days including tests (batch.py ~250 lines, CLI
wiring, validation tests, one end-to-end test with a fixture zip).
