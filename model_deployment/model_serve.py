#!/usr/bin/env python3
# serve_vllm.py — config-driven vLLM launcher (docker or native), OpenAI-compatible
# Usage:
#   python serve_vllm.py --config serve.yaml                   # uses YAML 'profile_to_run'
#   python serve_vllm.py --config serve.yaml --profile prod-h100
#
# Requirements: PyYAML (pip install pyyaml)
"""
Example curl commands (after starting the server):
  
no
"""

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

try:
    import yaml
except ImportError:
    print("Missing dependency: PyYAML. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)


# -----------------------------
# Simple logger
# -----------------------------
DEBUG = True

def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg: str) -> None:
    if DEBUG:
        print(f"[{_ts()}] {msg}", flush=True)


# -----------------------------
# YAML loading & helpers
# -----------------------------
def load_config(path: str) -> Dict[str, Any]:
    log(f"load_config() called with path={path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    log(f"load_config() loaded keys: {list(cfg.keys())}")
    return cfg


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    log(f"deep_merge() called:\n  base keys={list((base or {}).keys())}\n  override keys={list((override or {}).keys())}")
    out = dict(base or {})
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            log(f"deep_merge(): recursing into key='{k}' (both dicts)")
            out[k] = deep_merge(out[k], v)
        else:
            log(f"deep_merge(): setting key='{k}' -> {type(v).__name__}")
            out[k] = v
    log(f"deep_merge() result keys={list(out.keys())}")
    return out


# -----------------------------
# Flag builder (data-driven)
# -----------------------------
def _to_flag(key: str, key_map: Optional[Dict[str, str]] = None) -> str:
    flag = f"--{key_map[key]}" if (key_map and key in key_map) else f"--{key.replace('_', '-')}"
    log(f"_to_flag(): key='{key}' -> flag='{flag}'")
    return flag


def _coerce_to_args(flag: str, value: Any) -> List[str]:
    """
    Convert (flag, value) to CLI args:
      True  -> ["--flag"]
      False/None -> []
      list/tuple -> ["--flag", "v1", "--flag", "v2", ...]
      scalar -> ["--flag", str(value)]
    """
    log(f"_coerce_to_args(): flag='{flag}', value_type={type(value).__name__}, value={value}")
    if value is None:
        return []
    if isinstance(value, bool):
        out = [flag] if value else []
        log(f"_coerce_to_args(): bool -> {out}")
        return out
    if isinstance(value, (list, tuple)):
        out: List[str] = []
        for v in value:
            if v is None or v is False:
                continue
            if v is True:
                out.append(flag)
            else:
                out.extend([flag, str(v)])
        log(f"_coerce_to_args(): list/tuple -> {out}")
        return out
    out = [flag, str(value)]
    log(f"_coerce_to_args(): scalar -> {out}")
    return out


def build_flags_from_section(
    section: Dict[str, Any],
    include_keys: Optional[Iterable[str]] = None,
    exclude_keys: Optional[Iterable[str]] = None,
    key_map: Optional[Dict[str, str]] = None,
) -> List[str]:
    log(f"build_flags_from_section(): include={include_keys}, exclude={exclude_keys}")
    args: List[str] = []
    inc = set(include_keys) if include_keys else None
    exc = set(exclude_keys) if exclude_keys else set()
    for k, v in (section or {}).items():
        if inc and k not in inc:
            log(f"build_flags_from_section(): skipping '{k}' (not in include)")
            continue
        if k in exc:
            log(f"build_flags_from_section(): skipping '{k}' (in exclude)")
            continue
        flag = _to_flag(k, key_map)
        coerced = _coerce_to_args(flag, v)
        log(f"build_flags_from_section(): key='{k}' -> args={coerced}")
        args.extend(coerced)
    log(f"build_flags_from_section(): final args={args}")
    return args


def append_extra_args(args: List[str], extra: Optional[List[str]]) -> List[str]:
    log(f"append_extra_args(): starting args len={len(args)}, extra={extra}")
    for item in extra or []:
        if isinstance(item, str) and item.strip():
            split = shlex.split(item)
            log(f"append_extra_args(): adding {split}")
            args.extend(split)
    log(f"append_extra_args(): final args len={len(args)}")
    return args


# -----------------------------
# vLLM command construction
# -----------------------------
def build_vllm_command(model: str, server: Dict[str, Any], engine: Dict[str, Any], extra: List[str]) -> List[str]:
    log(f"build_vllm_command(): model={model}, server={server}")
    host = server.get("host", "0.0.0.0")
    port = int(server.get("port", 8000))
    base = ["--model", model, "--host", host, "--port", str(port)]
    log(f"build_vllm_command(): base args={base}")

    key_map = {
        # Uncomment if you need explicit override names:
        # "kv_cache_dtype": "kv-cache-dtype",
        # "tensor_parallel_size": "tensor-parallel-size",
        # "gpu_memory_utilization": "gpu-memory-utilization",
    }

    log(f"build_vllm_command(): engine raw={json.dumps(engine, indent=2)}")
    engine_flags = build_flags_from_section(engine, key_map=key_map)
    cmd = append_extra_args(base + engine_flags, extra)
    log(f"build_vllm_command(): final cmd fragment={cmd}")
    return cmd


# -----------------------------
# Readiness & test helpers
# -----------------------------
def poll_ready(base_url: str, timeout_s: int = 900, interval_s: float = 2.0) -> None:
    log(f"poll_ready(): base_url={base_url}, timeout_s={timeout_s}, interval_s={interval_s}")
    url = base_url.rstrip("/") + "/v1/models"
    last_err = None
    deadline = time.time() + timeout_s
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            log(f"poll_ready(): attempt={attempt}, GET {url}")
            with urllib.request.urlopen(url, timeout=5) as resp:
                body = resp.read().decode("utf-8")
                log(f"poll_ready(): status={resp.status}, body_len={len(body)}")
                if resp.status == 200:
                    data = json.loads(body)
                    if isinstance(data, dict) and "data" in data:
                        log("poll_ready(): server is ready ✅")
                        return
        except Exception as e:
            last_err = e
            log(f"poll_ready(): error on attempt {attempt}: {e}")
        time.sleep(interval_s)
    raise TimeoutError(f"Readiness timed out after {timeout_s}s; last error: {last_err}")


def test_chat(base_url: str, model: str, prompt: str, max_tokens: int = 32) -> str:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    log(f"test_chat(): POST {url}\n  payload={json.dumps(payload, indent=2)}")
    req = urllib.request.Request(url, data=data, headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-local-demo",
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            out = resp.read().decode("utf-8")
            log(f"test_chat(): status={resp.status}, resp_len={len(out)}")
            return out
    except urllib.error.HTTPError as he:
        body = he.read().decode("utf-8", errors="ignore")
        log(f"test_chat(): HTTPError {he.code} — {body[:500]}")
        raise
    except Exception as e:
        log(f"test_chat(): error {e}")
        raise


# -----------------------------
# Launcher (docker or native)
# -----------------------------
def run_process(cmd: List[str]) -> subprocess.Popen:
    printable = " ".join(shlex.quote(c) for c in cmd)
    log(f"run_process(): launching command:\n  {printable}")
    return subprocess.Popen(cmd)


def launch_from_config(cfg: Dict[str, Any], override_profile: Optional[str]) -> None:
    log(f"launch_from_config(): override_profile={override_profile}")
    profiles = cfg.get("profiles", {}) or {}
    chosen = override_profile or cfg.get("profile_to_run")
    log(f"launch_from_config(): available profiles={list(profiles.keys())}, chosen={chosen}")
    if not chosen or chosen not in profiles:
        raise SystemExit(f"Profile '{chosen}' not found (available: {', '.join(profiles.keys()) or 'none'}).")

    merged = deep_merge(cfg.get("defaults", {}), profiles[chosen])
    log(f"launch_from_config(): merged profile=\n{json.dumps(merged, indent=2)}")

    backend = merged.get("backend", "docker")
    image = merged.get("image", "vllm/vllm-openai:gptoss")
    host = merged.get("host", "0.0.0.0")
    port = int(merged.get("port", 8000))
    model = merged.get("model_id") or merged.get("model") or "openai/gpt-oss-20b"
    readiness_timeout = int(merged.get("readiness_timeout", 900))
    do_test = bool(merged.get("test_request", True))
    test_prompt = merged.get("test_prompt", "Say hello in one short sentence.")
    test_max_tokens = int(merged.get("test_max_tokens", 32))
    engine = merged.get("engine", {}) or {}
    extra_args = (cfg.get("extra_args") or []) + (merged.get("extra_args") or [])

    log(f"launch_from_config(): backend={backend}, image={image}, host={host}, port={port}")
    log(f"launch_from_config(): model={model}, readiness_timeout={readiness_timeout}, do_test={do_test}")
    log(f"launch_from_config(): test_prompt='{test_prompt}', test_max_tokens={test_max_tokens}")
    log(f"launch_from_config(): engine keys={list(engine.keys())}")
    log(f"launch_from_config(): extra_args={extra_args}")

    vllm_flags = build_vllm_command(model, {"host": host, "port": port}, engine, extra_args)
    log(f"launch_from_config(): vLLM flags={vllm_flags}")

    proc = None
    base_url = f"http://127.0.0.1:{port}"
    try:
        if backend == "docker":
            cmd = [
                "docker", "run", "--rm", "-it", "--gpus", "all",
                "-p", f"{port}:8000",
                "--ipc=host",
                image,
            ] + vllm_flags
            print(f"Going to run the docker command: {cmd}")
            proc = run_process(cmd)
        else:
            raise SystemExit(f"Unsupported backend: {backend}")

        log(f"launch_from_config(): waiting for readiness on {base_url}")
        poll_ready(base_url, timeout_s=readiness_timeout)
        log("launch_from_config(): server is ready ✅")

        if do_test:
            log("launch_from_config(): sending test request ...")
            out = test_chat(base_url, model=model, prompt=test_prompt, max_tokens=test_max_tokens)
            try:
                parsed = json.loads(out)
                preview = json.dumps(parsed, indent=2)[:2000]
                log(f"launch_from_config(): test response (JSON, truncated):\n{preview}")
            except json.JSONDecodeError:
                log(f"launch_from_config(): test response (raw, truncated):\n{out[:1000]}")

        print("\nQuick curl:")
        print(f"  curl {base_url}/v1/models -H 'Authorization: Bearer sk-local-demo'")
        print(f"  curl {base_url}/v1/chat/completions \\")
        print("    -H 'Content-Type: application/json' -H 'Authorization: Bearer sk-local-demo' \\")
        print(f"    -d '{{\"model\":\"{model}\",\"messages\":[{{\"role\":\"user\",\"content\":\"Hi\"}}],\"max_tokens\":16}}'")
        print("\nPress Ctrl+C to stop.")

        proc.wait()
        log("launch_from_config(): process exited")
    except KeyboardInterrupt:
        log("launch_from_config(): KeyboardInterrupt received, shutting down ...")
    finally:
        if proc and proc.poll() is None:
            try:
                log("launch_from_config(): sending SIGINT to child process ...")
                proc.send_signal(signal.SIGINT)
                time.sleep(2)
            except Exception as e1:
                log(f"launch_from_config(): SIGINT failed: {e1}; trying terminate()")
                try:
                    proc.terminate()
                except Exception as e2:
                    log(f"launch_from_config(): terminate() failed: {e2}")


def main():
    log("main(): starting")
    ap = argparse.ArgumentParser(description="Config-driven vLLM server launcher")
    ap.add_argument("--config", "-c", required=True, help="Path to YAML config (e.g., serve.yaml)")
    ap.add_argument("--profile", "-p", default=None, help="Profile name to run (overrides YAML profile_to_run)")
    args = ap.parse_args()
    log(f"main(): args.config={args.config}, args.profile={args.profile}")

    cfg = load_config(args.config)
    log("main(): loaded config (preview):\n" + json.dumps(cfg, indent=2)[:2000])
    launch_from_config(cfg, override_profile=args.profile)
    log("main(): done")


if __name__ == "__main__":
    main()