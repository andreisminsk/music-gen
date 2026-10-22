#!/usr/bin/env python3
"""Test RunPod pod lifecycle (deploy -> start -> stop -> terminate) on a CPU pod.

Uses the small `uhu-ollama-cpu` template (~$0.001-0.01/hr) so the whole test
costs a fraction of a cent. Exercises the same code paths as
runpod_remote_batch.py: REST v1 lifecycle, GraphQL fallbacks, runtime polling.

Usage:
    export RUNPOD_API_KEY=...
    python scripts/test_pod_lifecycle.py            # full lifecycle
    python scripts/test_pod_lifecycle.py --keep     # leave the pod stopped (no terminate)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

API_GRAPHQL = "https://api.runpod.io/graphql"
API_REST = "https://rest.runpod.io/v1"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
TEMPLATE_NAME = "uhu-ollama-cpu"
READY_TIMEOUT = 300  # CPU pods boot fast; small image


def log(msg: str) -> None:
    print(msg, flush=True)


def die(msg: str, code: int = 1) -> None:
    print(f"ERROR: {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


def _request(url: str, data: bytes | None = None, headers: dict | None = None,
             method: str = "GET") -> dict:
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = r.read().decode()
            return json.loads(body) if body.strip() else {}
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {"_http": e.code, "_body": body[:300]}


def gql(query: str) -> dict:
    """GraphQL POST; raises RuntimeError on GraphQL errors."""
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        die("RUNPOD_API_KEY is not set")
    data = _request(
        f"{API_GRAPHQL}?api_key={api_key}",
        data=json.dumps({"query": query}).encode(),
        headers={"Content-Type": "application/json", "User-Agent": UA},
        method="POST",
    )
    if data.get("errors"):
        raise RuntimeError(json.dumps(data["errors"])[:300])
    return data.get("data") or {}


def rest(method: str, path: str, body: dict | None = None) -> dict:
    """REST v1 call with Bearer auth."""
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        die("RUNPOD_API_KEY is not set")
    headers = {"Authorization": f"Bearer {api_key}", "User-Agent": UA}
    data = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body).encode()
    return _request(f"{API_REST}{path}", data=data, headers=headers, method=method)


def resolve_template(name: str) -> str:
    for t in gql("query { myself { podTemplates { id name } } }")["myself"]["podTemplates"]:
        if t["name"] == name:
            return t["id"]
    die(f"Template {name!r} not found on this account")


def get_pod(pod_id: str) -> dict | None:
    q = f'query {{ pod(input: {{ podId: "{pod_id}" }}) {{ id desiredStatus ' \
        f'runtime {{ uptimeInSeconds ports {{ ip privatePort publicPort }} }} }} }}'
    return gql(q).get("pod")


def wait_runtime(pod_id: str, timeout_s: int) -> bool:
    """Wait until the pod reports a runtime (container booted)."""
    deadline = time.time() + timeout_s
    waited = 0
    while time.time() < deadline:
        pod = get_pod(pod_id)
        rt = (pod or {}).get("runtime")
        if rt is not None:
            return True
        if waited and waited % 60 == 0:
            log(f"    still booting ({waited}s)...")
        time.sleep(5)
        waited += 5
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Test pod lifecycle on a cheap CPU pod")
    ap.add_argument("--template", default=TEMPLATE_NAME, help=f"Template name (default: {TEMPLATE_NAME})")
    ap.add_argument("--keep", action="store_true",
                    help="Leave the pod stopped at the end (default: terminate)")
    ap.add_argument("--ready-timeout", type=int, default=READY_TIMEOUT)
    args = ap.parse_args()

    results: list[tuple[str, bool, str]] = []

    def check(step: str, ok: bool, detail: str = "") -> None:
        results.append((step, ok, detail))
        log(f"  [{'PASS' if ok else 'FAIL'}] {step}" + (f" — {detail}" if detail else ""))

    template_id = resolve_template(args.template)
    log(f"Template: {args.template} -> {template_id}")

    pod_id = None
    try:
        # --- 1. deploy (CPU pod via REST v1) ---
        log("[1/5] Deploy CPU pod...")
        try:
            r = rest("POST", "/pods", body={
                "name": f"lifecycle-test-{int(time.time())}",
                "templateId": template_id,
                "computeType": "CPU",
                "cloudType": "SECURE",
                "containerDiskInGb": 10,
            })
            pod_id = r.get("id")
            check("deploy (REST POST /pods, computeType=CPU)", bool(pod_id),
                  f"pod id: {pod_id}" if pod_id else f"unexpected response: {json.dumps(r)[:200]}")
        except (RuntimeError, urllib.error.URLError) as e:
            check("deploy (REST POST /pods, computeType=CPU)", False, str(e)[:200])

        if not pod_id:
            die("Cannot continue without a pod")

        # --- 2. runtime comes up ---
        log("[2/5] Wait for runtime (container boot)...")
        ok = wait_runtime(pod_id, args.ready_timeout)
        pod = get_pod(pod_id) or {}
        check("runtime up", ok, f"desired={pod.get('desiredStatus')}")

        # --- 3. stop ---
        log("[3/5] Stop pod (REST, GraphQL fallback)...")
        stopped = False
        try:
            rest("POST", f"/pods/{pod_id}/stop")
            stopped = True
        except (RuntimeError, urllib.error.URLError) as e:
            log(f"    REST stop failed ({e}); trying GraphQL podStop...")
            try:
                gql(f'mutation {{ podStop(input: {{ podId: "{pod_id}" }}) {{ id desiredStatus }} }}')
                stopped = True
            except (RuntimeError, urllib.error.URLError) as e2:
                check("stop", False, f"REST: {e}; GraphQL: {e2}")
        if stopped:
            time.sleep(5)
            pod = get_pod(pod_id) or {}
            check("stop", pod.get("desiredStatus") in ("EXITED", "STOPPED"),
                  f"desired={pod.get('desiredStatus')}")

        # --- 4. start again ---
        log("[4/5] Start pod again...")
        started = False
        try:
            rest("POST", f"/pods/{pod_id}/start")
            started = True
        except (RuntimeError, urllib.error.URLError) as e:
            log(f"    REST start failed ({e}); trying GraphQL podStart...")
            try:
                gql(f'mutation {{ podStart(input: {{ podId: "{pod_id}" }}) {{ id desiredStatus }} }}')
                started = True
            except (RuntimeError, urllib.error.URLError) as e2:
                check("start", False, f"REST: {e}; GraphQL: {e2}")
        if started:
            # Poll for the actual state transition EXITED -> RUNNING; a stopped
            # pod can report stale runtime data immediately, so waiting for
            # runtime alone is not enough (that's how the first run failed).
            deadline = time.time() + args.ready_timeout
            desired = None
            while time.time() < deadline:
                pod = get_pod(pod_id) or {}
                desired = pod.get("desiredStatus")
                if desired == "RUNNING":
                    break
                time.sleep(5)
            check("start", desired == "RUNNING", f"desired={desired}")

        # --- 5. stop + terminate ---
        log("[5/5] Stop, then terminate...")
        try:
            rest("POST", f"/pods/{pod_id}/stop")
        except (RuntimeError, urllib.error.URLError) as e:
            log(f"    REST stop failed ({e}); trying GraphQL podStop...")
            gql(f'mutation {{ podStop(input: {{ podId: "{pod_id}" }}) {{ id desiredStatus }} }}')
        time.sleep(5)
        pod = get_pod(pod_id) or {}
        check("stop before terminate", pod.get("desiredStatus") in ("EXITED", "STOPPED"),
              f"desired={pod.get('desiredStatus')}")

        if args.keep:
            log("    --keep set: leaving the pod stopped (storage billing applies).")
            results.append(("terminate", True, "skipped (--keep)"))
        else:
            terminated = False
            try:
                rest("DELETE", f"/pods/{pod_id}")
                terminated = True
            except (RuntimeError, urllib.error.URLError) as e:
                log(f"    REST terminate failed ({e}); trying GraphQL podTerminate...")
                try:
                    gql(f'mutation {{ podTerminate(input: {{ podId: "{pod_id}" }}) }}')
                    terminated = True
                except (RuntimeError, urllib.error.URLError) as e2:
                    check("terminate", False, f"REST: {e}; GraphQL: {e2}")
            if terminated:
                time.sleep(5)
                gone = get_pod(pod_id) is None
                check("terminate", gone, "pod no longer exists" if gone else "pod still visible")

    finally:
        # Safety net: never leave a running pod behind
        if pod_id and not args.keep:
            pod = get_pod(pod_id)
            if pod is not None:
                log("Safety net: terminating leftover pod...")
                try:
                    rest("DELETE", f"/pods/{pod_id}")
                except (RuntimeError, urllib.error.URLError):
                    try:
                        gql(f'mutation {{ podTerminate(input: {{ podId: "{pod_id}" }}) }}')
                    except (RuntimeError, urllib.error.URLError) as e:
                        log(f"    WARNING: terminate failed ({e}) — HANDLE IN CONSOLE")

    log("\n" + "=" * 52)
    passed = sum(1 for _, ok, _ in results if ok)
    for step, ok, detail in results:
        log(f"  [{'PASS' if ok else 'FAIL'}] {step}" + (f" ({detail})" if detail else ""))
    log(f"= {passed}/{len(results)} checks passed")
    log("=" * 52)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
