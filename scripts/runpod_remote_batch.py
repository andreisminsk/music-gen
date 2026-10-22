#!/usr/bin/env python3
"""Run a music-gen batch on a RunPod pod — one command, with cost guardrails.

Cross-platform (Windows 11 PowerShell / macOS / Linux). Replaces a .sh/.ps1
pair: ssh and scp are built into Win11 (OpenSSH client optional feature) and
macOS, and Python 3.10+ is required by music-gen anyway.

Flow:
  balance before -> cheapest compatible GPU (user confirms) -> deploy or start
  pod -> wait for SSH -> upload batch zip -> music-gen batch (streamed) ->
  download output -> stop pod (ALWAYS, even on Ctrl+C) -> balance after
  (billing-lag aware, with an independent cost estimate).

Requirements:
  RUNPOD_API_KEY env var (https://www.runpod.io/console/user/settings)
  ssh/scp in PATH; a pod template with music-gen installed
  (default template name: image-gen1.0.0-cu128).

Usage:
  python runpod_remote_batch.py batch.zip
  python runpod_remote_batch.py batch_dir/ --yes
  python runpod_remote_batch.py batch.zip --pod-id <existing-pod-id>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

API_GRAPHQL = "https://api.runpod.io/graphql"
API_REST = "https://rest.runpod.io/v1"
# Cloudflare blocks the default urllib User-Agent (error 1010) — use a browser UA.
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
DEFAULT_TEMPLATE = "YuE2-3B-music-gen-0.2.0-torch2.10.0-cu128"
MIN_VRAM_DEFAULT = 24  # YuE2-3B requires 24 GB (BF16)


def log(msg: str) -> None:
    print(msg, flush=True)


def die(msg: str, code: int = 1) -> None:
    print(f"ERROR: {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


# ---------------------------------------------------------------- RunPod API


def gql(query: str) -> dict:
    """POST a GraphQL query; returns data dict, raises RuntimeError on errors."""
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        die("RUNPOD_API_KEY is not set (https://www.runpod.io/console/user/settings)")
    req = urllib.request.Request(
        f"{API_GRAPHQL}?api_key={api_key}",
        data=json.dumps({"query": query}).encode(),
        headers={"Content-Type": "application/json", "User-Agent": UA},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        die(f"GraphQL HTTP {e.code}: {e.read().decode(errors='replace')[:300]}")
    except urllib.error.URLError as e:
        die(f"GraphQL connection failed: {e.reason}")
    if data.get("errors"):
        raise RuntimeError(json.dumps(data["errors"])[:300])
    return data["data"]


def rest(action: str, pod_id: str) -> None:
    """POST to the v1 REST lifecycle endpoint (start/stop) with Bearer auth."""
    api_key = os.environ.get("RUNPOD_API_KEY")
    req = urllib.request.Request(
        f"{API_REST}/pods/{pod_id}/{action}",
        data=b"",
        headers={"Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30):
        pass


def get_balance() -> tuple[float, float, object]:
    d = gql("query { myself { clientBalance currentSpendPerHr spendLimit } }")["myself"]
    return (float(d["clientBalance"] or 0),
            float(d["currentSpendPerHr"] or 0),
            d["spendLimit"])


def pick_gpu(args) -> dict:
    """Return the confirmed GPU type. Offers the cheapest >= min VRAM first."""
    data = gql(
        "query { gpuTypes { id displayName memoryInGb "
        "lowestPrice(input: { gpuCount: 1 }) { uninterruptablePrice } } }"
    )["gpuTypes"]
    if args.gpu:
        for g in data:
            if g["id"] == args.gpu:
                return g
        die(f"GPU id {args.gpu!r} not found (see https://docs.runpod.io/references/gpu-types)")

    cands = sorted(
        (g for g in data
         if g["memoryInGb"] >= args.vram
         and (g.get("lowestPrice") or {}).get("uninterruptablePrice")),
        key=lambda g: g["lowestPrice"]["uninterruptablePrice"],
    )
    if not cands:
        die(f"No GPU with >= {args.vram} GB VRAM currently available")

    top = cands[:5]
    log(f"\nCheapest compatible GPUs (>={args.vram} GB VRAM):")
    for i, g in enumerate(top, 1):
        log(f"  [{i}] {g['displayName']:<30} {g['memoryInGb']:>3} GB  "
            f"${g['lowestPrice']['uninterruptablePrice']:.2f}/hr")
    if args.yes:
        log(f"  --yes: auto-selected [1] {top[0]['displayName']}")
        choice = 1
    else:
        raw = input(f"Deploy on [1-{len(top)}], Enter=cheapest, q=quit: ").strip().lower()
        if raw == "q":
            die("Aborted by user", 0)
        choice = int(raw) if raw.isdigit() and 1 <= int(raw) <= len(top) else 1
    return top[choice - 1]


def resolve_template(name: str, template_id: str | None) -> str:
    if template_id:
        return template_id
    try:
        templates = gql("query { myself { podTemplates { id name } } }")["myself"]["podTemplates"]
    except RuntimeError as e:
        die(f"Cannot resolve template {name!r} by name ({e}). "
            f"Pass --template-id explicitly (find it in RunPod console > Templates).")
    for t in templates:
        if t["name"] == name:
            return t["id"]
    die(f"Template {name!r} not found on this account. "
        f"Pass --template-id explicitly.")


def deploy_pod(gpu: dict, template_id: str, args) -> str:
    inp = (
        "cloudType: ALL, gpuCount: 1, "
        f'gpuTypeId: "{gpu["id"]}", '
        f"containerDiskInGb: {args.disk}, volumeInGb: {args.volume}, "
        "minVcpuCount: 4, minMemoryInGb: 16, "
        f'name: "music-gen-batch-{int(time.time())}", '
        f'templateId: "{template_id}", '
        'ports: "22/tcp", volumeMountPath: "/workspace"'
    )
    d = gql("mutation { podFindAndDeployOnDemand(input: { " + inp + " }) { id } }")
    pod = d.get("podFindAndDeployOnDemand")
    if not pod or not pod.get("id"):
        die(f"Deployment failed (no pod id returned): {d}")
    return pod["id"]


def wait_runtime(pod_id: str, timeout_s: int) -> tuple[str, int]:
    """Poll until the pod exposes SSH (private 22 -> public port). Returns (ip, port)."""
    q = (f'query {{ pod(input: {{ podId: "{pod_id}" }}) {{ runtime {{ uptimeInSeconds '
         f'ports {{ ip privatePort publicPort }} }} }} }}')
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            pod = gql(q)["pod"]
            rt = (pod or {}).get("runtime")
            for p in (rt or {}).get("ports") or []:
                if p["privatePort"] == 22 and p.get("publicPort") and p.get("ip"):
                    return p["ip"], int(p["publicPort"])
        except RuntimeError:
            pass  # pod not registered yet
        time.sleep(10)
    die(f"Pod {pod_id} did not become ready within {timeout_s}s")


def wait_ssh(ip: str, port: int, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection((ip, port), timeout=5):
                time.sleep(3)  # let sshd finish initializing
                return
        except OSError:
            time.sleep(5)
    die(f"SSH port {ip}:{port} not reachable within {timeout_s}s")


def stop_pod(pod_id: str) -> None:
    rest("stop", pod_id)


# ------------------------------------------------------------------- helpers


def zip_batch(src: Path) -> Path:
    """Return a zip path: pass files through, zip directories on the fly."""
    if src.is_file():
        if src.suffix.lower() != ".zip":
            die(f"Batch source must be a .zip or a directory containing job.json: {src}")
        return src
    if not (src / "job.json").is_file():
        die(f"No job.json in {src} — not a valid batch directory")
    tmp = Path(tempfile.mkdtemp(prefix="runpod-batch-")) / "batch.zip"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(src.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(src))
    return tmp


def ssh_opts(port: int) -> list[str]:
    return ["-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=15",
            "-p", str(port)]


def scp_opts(port: int) -> list[str]:
    return ["-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=15",
            "-P", str(port)]  # scp uses capital -P


def wait_settled(before: float, timeout_s: int) -> tuple[float, bool]:
    """Billing-lag mitigation.

    Phase 1: poll until currentSpendPerHr == 0 — confirms the GPU is actually
    released and nothing is accruing anymore.
    Phase 2: poll the balance for up to timeout_s waiting for the session
    charge to post. Returns (after_balance, settled?).
    """
    deadline = time.time() + 120
    while time.time() < deadline:
        _, spend, _ = get_balance()
        if spend == 0:
            break
        time.sleep(10)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        after, _, _ = get_balance()
        if abs(after - before) > 0.004:  # balance moved — charge posted
            return after, True
        time.sleep(15)
    after, _, _ = get_balance()
    return after, False


# ---------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run a music-gen batch on a RunPod pod with cost guardrails")
    ap.add_argument("source", nargs="?", default=None,
                   help="Batch zip or directory containing job.json")
    ap.add_argument("--pod-id", help="Reuse an existing (stopped) pod instead of deploying")
    ap.add_argument("--template", default=DEFAULT_TEMPLATE,
                    help=f"Pod template name (default: {DEFAULT_TEMPLATE})")
    ap.add_argument("--template-id", help="Pod template ID (skips name lookup)")
    ap.add_argument("--gpu", help="Explicit GPU type id (skips cheapest-GPU prompt)")
    ap.add_argument("--vram", type=int, default=MIN_VRAM_DEFAULT,
                    help=f"Minimum VRAM in GB (default: {MIN_VRAM_DEFAULT})")
    ap.add_argument("--disk", type=int, default=50, help="Container disk GB (default: 50)")
    ap.add_argument("--volume", type=int, default=10, help="Volume disk GB (default: 10)")
    ap.add_argument("--remote-dir", default="/workspace", help="Remote work dir")
    ap.add_argument("--output", help="Local dir for downloaded results "
                                     "(default: output-<timestamp>)")
    ap.add_argument("--yes", action="store_true", help="Non-interactive: accept cheapest GPU")
    ap.add_argument("--keep-pod", action="store_true",
                    help="Do NOT stop the pod at the end (billed hourly!)")
    ap.add_argument("--ready-timeout", type=int, default=900,
                    help="Pod boot timeout s (default: 900)")
    ap.add_argument("--ssh-timeout", type=int, default=300, help="SSH reach timeout s")
    ap.add_argument("--billing-wait", type=int, default=300,
                    help="Max s to wait for the balance to post the charge (default: 300)")
    ap.add_argument("--balance-only", action="store_true",
                    help="Print the current balance and exit (no pod, no batch)")
    args = ap.parse_args()

    if args.balance_only:
        bal, spend, limit = get_balance()
        log(f"Balance:     ${bal:.2f}")
        log(f"Spend/hr:    ${spend:.2f}")
        log(f"Spend limit: {limit}")
        return 0

    if not args.source:
        die("Batch source is required (zip or directory containing job.json), "
            "or use --balance-only")

    for tool in ("ssh", "scp"):
        if not shutil.which(tool):
            die(f"{tool!r} not in PATH — Win11: Settings > Apps > Optional features > "
                f"OpenSSH Client; macOS/Linux: preinstalled")

    src = Path(args.source)
    if not src.exists():
        die(f"Batch source not found: {src}")
    zip_path = zip_batch(src)

    # [1/6] balance before
    bal_before, spend_hr, limit = get_balance()
    log(f"[1/6] Balance before: ${bal_before:.2f}  (spend/hr ${spend_hr:.2f}, limit {limit})")

    # [2/6] GPU + pod
    gpu = pick_gpu(args)
    template_id = resolve_template(args.template, args.template_id)
    if args.pod_id:
        pod_id = args.pod_id
        log(f"[2/6] Reusing pod {pod_id} — starting...")
        rest("start", pod_id)
    else:
        pod_id = deploy_pod(gpu, template_id, args)
        log(f"[2/6] Pod deployed: {pod_id}  "
            f"({gpu['displayName']} @ ${gpu['lowestPrice']['uninterruptablePrice']:.2f}/hr)")
    t0 = time.time()

    failure: Exception | None = None
    remote = args.remote_dir.rstrip("/")
    try:
        ip, port = wait_runtime(pod_id, args.ready_timeout)
        log(f"      SSH endpoint: root@{ip}:{port}")
        wait_ssh(ip, port, args.ssh_timeout)

        log(f"[3/6] Uploading {zip_path.name} ({zip_path.stat().st_size / 1e6:.1f} MB)...")
        subprocess.run(["scp", *scp_opts(port), str(zip_path),
                        f"root@{ip}:{remote}/batch.zip"], check=True)

        log("[4/6] Running music-gen batch (streaming)...")
        rc = subprocess.run(
            ["ssh", *ssh_opts(port), f"root@{ip}",
             f"cd {remote} && music-gen batch --zip {remote}/batch.zip -o {remote}/output"],
            check=False).returncode

        local_out = Path(args.output) if args.output else \
            Path(f"output-{datetime.now():%Y%m%d-%H%M%S}")
        log(f"[5/6] Downloading results to {local_out} ...")
        subprocess.run(["scp", *scp_opts(port), "-r",
                        f"root@{ip}:{remote}/output", str(local_out)], check=True)
        log(f"      Downloaded: {local_out}")
    except (subprocess.CalledProcessError, TimeoutError, OSError) as e:
        failure = e
        log(f"ERROR: {e}")
    except KeyboardInterrupt:
        failure = KeyboardInterrupt()
        log("Interrupted — stopping pod before exit.")
    finally:
        gpu_seconds = time.time() - t0
        # [6/6] ALWAYS stop — the forgotten-pod risk from SERVERLESS-RUNPOD.md §12
        if args.keep_pod:
            log(f"[6/6] WARNING: --keep-pod set — pod {pod_id} is STILL RUNNING (billed hourly).")
        else:
            log(f"[6/6] Stopping pod {pod_id} ...")
            try:
                stop_pod(pod_id)
            except Exception as e:  # noqa: BLE001 — must not mask the original error
                log(f"WARNING: pod stop failed ({e}) — STOP IT MANUALLY in the RunPod "
                    f"console to avoid idle charges!")

    # Balance after — billing-lag aware
    est = None
    if not args.pod_id:
        est = gpu["lowestPrice"]["uninterruptablePrice"] * gpu_seconds / 3600
    after, settled = wait_settled(bal_before, args.billing_wait)
    delta = bal_before - after

    log("\n" + "=" * 52)
    log(f"Balance before:  ${bal_before:.2f}")
    log(f"Balance after:   ${after:.2f}"
        + ("" if settled else "   (not settled yet)"))
    log(f"Session cost:    ${delta:.2f}"
        + ("" if settled else "   (partial — RunPod posts billing asynchronously)"))
    if est is not None:
        log(f"GPU estimate:    ${est:.2f}   ({gpu['displayName']} @ "
            f"${gpu['lowestPrice']['uninterruptablePrice']:.2f}/hr x {gpu_seconds / 60:.0f} min)")
    if not settled:
        log("NOTE: re-run `python runpod_remote_batch.py --balance-only` in a few "
            "minutes for the final number.")
    log("=" * 52)

    if failure:
        return 1
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
