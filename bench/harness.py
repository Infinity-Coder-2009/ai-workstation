#!/usr/bin/env python3
"""Benchmark harness for the pilot box.

Drives any OpenAI-compatible endpoint (llama.cpp server, our runtime, vLLM)
and measures TTFT, prefill tok/s, decode tok/s per model/case. Optional soak
mode + thermals/power/SSD-endurance sampling.

Examples:
  # serve a model with llama.cpp, then bench it:
  python harness.py --endpoint http://127.0.0.1:8080 --model /models/kimi-k2-q4.gguf
  # or let the harness launch llama-server itself:
  python harness.py --llama-server ./llama-server --gguf /models/kimi-k2-q4.gguf --ctx 65536
  # soak test (8h) with telemetry:
  python harness.py --endpoint http://127.0.0.1:8080 --model m --soak-hours 8 --telemetry telemetry.csv
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
from models import MODELS, CASES  # noqa: E402


def wait_health(endpoint: str, timeout: float = 120.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(endpoint + "/health", timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"endpoint {endpoint} never became healthy")


def chat_stream(endpoint: str, model: str, prompt: str, max_tokens: int):
    """Returns (ttft_s, prefill_toks, decode_toks, wall_s). Streams SSE."""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        endpoint + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    decode = 0
    usage = {}
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if ttft is None and chunk.get("choices") and chunk["choices"][0].get("delta", {}).get("content"):
                ttft = time.time() - t0
            if chunk.get("choices") and chunk["choices"][0].get("delta", {}).get("content"):
                decode += 1
            if chunk.get("usage"):
                usage = chunk["usage"]
    wall = time.time() - t0
    prefill = usage.get("prompt_tokens", 0)
    return ttft, prefill, decode, wall


def make_prompt(target_tokens: int) -> str:
    # ~0.75 tokens/word; repeat a code-ish block to approximate prompt length
    unit = ("def process_batch(items):\n"
            "    results = []\n"
            "    for item in items:\n"
            "        results.append(transform(item))\n"
            "    return results\n\n") * 20
    words = int(target_tokens / 0.75 / 6)
    return (unit * max(1, words))[: int(target_tokens * 4)]


def run_case(endpoint, model, case_name, prompt_tokens, gen_tokens):
    prompt = make_prompt(prompt_tokens) + "\nSummarize the code above in one sentence."
    ttft, prefill, decode, wall = chat_stream(endpoint, model, prompt, gen_tokens)
    return {
        "model": model, "case": case_name,
        "prompt_tokens": prefill, "gen_tokens": decode,
        "ttft_s": round(ttft, 2) if ttft else None,
        "prefill_tok_s": round(prefill / ttft, 1) if ttft and prefill else None,
        "decode_tok_s": round(decode / (wall - ttft), 1) if ttft and wall > ttft else None,
        "wall_s": round(wall, 1),
    }


def sample_telemetry():
    row = {"t": time.time()}
    if shutil.which("nvidia-smi"):
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,utilization.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
        if out:
            temp, power, util = out.split(", ")
            row.update(gpu_temp_c=float(temp), gpu_power_w=float(power), gpu_util=float(util))
    if shutil.which("smartctl"):
        out = subprocess.run(["smartctl", "-A", "/dev/nvme0"], capture_output=True, text=True).stdout
        for ln in out.splitlines():
            if "Percentage Used" in ln:
                row["nvme_pct_used"] = int(ln.split(":")[-1].strip().rstrip("%"))
    return row


def telemetry_loop(path, stop: threading.Event, interval=30):
    with open(path, "w") as f:
        f.write("timestamp,gpu_temp_c,gpu_power_w,gpu_util,nvme_pct_used\n")
        while not stop.is_set():
            row = sample_telemetry()
            f.write(",".join(str(row.get(k, "")) for k in
                     ("t", "gpu_temp_c", "gpu_power_w", "gpu_util", "nvme_pct_used")) + "\n")
            f.flush()
            stop.wait(interval)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", help="OpenAI-compatible base URL (e.g. http://127.0.0.1:8080)")
    ap.add_argument("--llama-server", help="path to llama-server binary to launch ourselves")
    ap.add_argument("--gguf", help="GGUF file for self-launched server")
    ap.add_argument("--ctx", type=int, default=65536)
    ap.add_argument("--ngl", type=int, default=99, help="GPU layers")
    ap.add_argument("--model", required=True, help="model id to send in requests")
    ap.add_argument("--cases", default="all", help="comma list or 'all'")
    ap.add_argument("--soak-hours", type=float, default=0)
    ap.add_argument("--telemetry", help="CSV path for telemetry during soak")
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    proc = None
    endpoint = args.endpoint
    if args.llama_server:
        cmd = [args.llama_server, "-m", args.gguf, "--ctx-size", str(args.ctx),
               "-ngl", str(args.ngl), "--host", "127.0.0.1", "--port", "8080",
               "--flash-attn", "on"]
        print("launching:", " ".join(cmd))
        proc = subprocess.Popen(cmd)
        endpoint = "http://127.0.0.1:8080"

    try:
        wait_health(endpoint)
        cases = CASES if args.cases == "all" else \
            [c for c in CASES if c[0] in args.cases.split(",")]
        results = []
        stop = threading.Event()
        tel = None
        if args.soak_hours > 0 and args.telemetry:
            tel = threading.Thread(target=telemetry_loop, args=(args.telemetry, stop), daemon=True)
            tel.start()

        deadline = time.time() + args.soak_hours * 3600 if args.soak_hours else None
        while True:
            for name, ptok, gtok in cases:
                r = run_case(endpoint, args.model, name, ptok, gtok)
                print(json.dumps(r))
                results.append(r)
            if deadline is None or time.time() >= deadline:
                break
        stop.set()
        with open(args.out, "w") as f:
            json.dump({"model": args.model, "endpoint": endpoint, "results": results}, f, indent=2)
        print(f"wrote {args.out} ({len(results)} runs)")
    finally:
        if proc:
            proc.terminate()


if __name__ == "__main__":
    main()
