#!/usr/bin/env python3
"""Runs every measurement in the README and writes results/*.json. Host-side: drives Docker.

    python3 scripts/run_experiments.py [--only latency,threads,load,micro,parity] [--quick]

CPU isolation: the system under test (edge_bench or edge_server) runs in a container pinned to
cores 0-3 (`--cpuset-cpus 0-3`, the "4-core device"); the load generator runs in a second container
pinned to cores 6-9, so client work never competes with the server for the cores being measured.
Everything uses the Release build in the `edge-infer-main` volume (built by this script).

Outputs:
  results/env.json                   CPU model, cores, RAM, compiler, flags, ORT version
  results/latency.json               per model: load time, p50/p95 per batch size, peak RSS
  results/threads.json               intra-op threads x concurrent callers (oversubscription), spinning on/off
  results/load.json                  open-loop load, batching on vs off, at multiples of capacity
  results/microbench.json            Google Benchmark: preprocessing kernel, mutex queue vs lock-free ring
  results/parity_*.json              C++ server vs Python ONNX Runtime on the same inputs
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
IMG = "edge-infer-dev"
VOL = "edge-infer-main"
SUT_CPUS = "0-3"
CLIENT_CPUS = "6-9"
NET = "edge-net"

MODELS = ["mobilenet_v3_large_fp32", "mobilenet_v3_large_int8", "resnet50_v2_fp32", "resnet50_v2_int8",
          "distilbert_sst2_fp32", "distilbert_sst2_int8"]


def sh(cmd: list[str], check: bool = True, capture: bool = False) -> str:
    r = subprocess.run(cmd, text=True, capture_output=capture)
    if check and r.returncode != 0:
        raise SystemExit(f"failed: {' '.join(cmd)}\n{r.stdout if capture else ''}{r.stderr if capture else ''}")
    return r.stdout if capture else ""


def drun(args: list[str], cpus: str = SUT_CPUS, name: str | None = None, detach: bool = False, capture: bool = True) -> str:
    cmd = ["docker", "run", "--rm", "--cpuset-cpus", cpus, "--network", NET, "-v", f"{ROOT}:/src", "-v", f"{VOL}:/src/build"]
    if name:
        cmd += ["--name", name]
    if detach:
        cmd += ["-d"]
    return sh(cmd + [IMG] + args, capture=capture)


def build() -> None:
    subprocess.run(["docker", "network", "create", NET], capture_output=True)
    drun(["bash", "-c", "cmake --preset release >/tmp/c.log 2>&1 || (tail -30 /tmp/c.log; exit 1); cmake --build --preset release 2>&1 | tail -2"],
         cpus="0-11", capture=False)


def env() -> None:
    script = r"""
import json, os, platform, subprocess, re
cpu = next((l.split(':',1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')), platform.machine())
flags = next((l.split(':',1)[1].split() for l in open('/proc/cpuinfo') if l.startswith('flags')), [])
mem = int(next(l.split()[1] for l in open('/proc/meminfo') if l.startswith('MemTotal'))) / 1024 / 1024
gcc = subprocess.run(['g++', '--version'], capture_output=True, text=True).stdout.splitlines()[0]
print(json.dumps({'cpu': cpu, 'arch': platform.machine(), 'avx2': 'avx2' in flags, 'logical_cpus_in_vm': os.cpu_count(),
  'mem_gib_in_vm': round(mem, 1), 'compiler': gcc, 'cxx_flags': '-O3 -DNDEBUG (Release preset); -mavx2 -mfma on the AVX2 unit only',
  'onnxruntime': '1.20.1 (official prebuilt, CPU execution provider)', 'os': open('/etc/os-release').read().split('PRETTY_NAME=')[1].split('\n')[0].strip('"'),
  'host': 'Docker Desktop VM on an Intel MacBook Pro (12 logical CPUs, 8 GiB to the VM)', 'sut_cpuset': '%s', 'client_cpuset': '%s'}, indent=2))
""" % (SUT_CPUS, CLIENT_CPUS)
    out = drun(["python3", "-c", script])
    (RES / "env.json").write_text(out)
    print(out)


def latency(quick: bool) -> None:
    rows = []
    for m in MODELS:
        path = f"models/{m}/model.onnx"
        if not (ROOT / path).exists():
            continue
        out = f"/src/results/.lat_{m}.json"
        drun(["./build/release/edge_bench", "latency", "--model", path, "--batches", "1,8", "--intra", "4",
              "--iters", "10" if quick else "40", "--out", out])
        row = json.loads((RES / f".lat_{m}.json").read_text())
        (RES / f".lat_{m}.json").unlink()
        row["model"] = m
        row["file_mib"] = round((ROOT / path).stat().st_size / 2**20, 1)
        rows.append(row)
        b1 = next(r for r in row["rows"] if r["batch"] == 1)
        print(f"  {m:26s} load {row['load_ms']:7.0f} ms  b1 p50 {b1['p50_ms']:7.2f} ms  peak RSS {row['peak_rss_mib']:7.1f} MiB")
    (RES / "latency.json").write_text(json.dumps({"intra_op_threads": 4, "cpuset": SUT_CPUS, "models": rows}, indent=2))


def threads(quick: bool) -> None:
    out = {}
    for m in ("mobilenet_v3_large_int8", "resnet50_v2_int8"):
        drun(["./build/release/edge_bench", "threads", "--model", f"models/{m}/model.onnx", "--intra", "1,2,4,8",
              "--callers", "1,2,4,8", "--seconds", "2" if quick else "4", "--spin", "--out", f"/src/results/.thr_{m}.json"])
        out[m] = json.loads((RES / f".thr_{m}.json").read_text())
        (RES / f".thr_{m}.json").unlink()
    (RES / "threads.json").write_text(json.dumps({"cpuset": SUT_CPUS, "cores": 4, "models": out}, indent=2))


def start_server(model: str, max_batch: int, wait_us: int, workers: int, intra: int) -> None:
    subprocess.run(["docker", "rm", "-f", "edge-sut"], capture_output=True)
    drun(["./build/release/edge_server", "--model", f"models/{model}/model.onnx", "--port", "8080", "--max-batch", str(max_batch),
          "--max-wait-us", str(wait_us), "--workers", str(workers), "--intra", str(intra), "--queue", "256"],
         name="edge-sut", detach=True)
    for _ in range(60):
        r = subprocess.run(["docker", "run", "--rm", "--network", NET, IMG, "curl", "-sf", "edge-sut:8080/health"], capture_output=True, text=True)
        if r.returncode == 0:
            return
        time.sleep(1)
    raise SystemExit("server did not come up")


def loadgen(rate: float, seconds: float, kind: str) -> dict:
    out = drun(["./build/release/edge_loadgen", "--host", "edge-sut", "--port", "8080", "--rate", str(rate), "--seconds", str(seconds),
                "--kind", kind, "--senders", "128", "--out", "/src/results/.lg.json"], cpus=CLIENT_CPUS)
    d = json.loads((RES / ".lg.json").read_text())
    (RES / ".lg.json").unlink()
    return d


def load(quick: bool) -> None:
    """Capacity is measured, not assumed: a 2x overload run on the batching-off server gives its
    sustained goodput, and the load levels are multiples of that."""
    seconds = 8 if quick else 20
    out = {"cpuset": SUT_CPUS, "seconds_per_point": seconds, "models": {}}
    # 4 workers x 1 intra-op thread: the best (callers x threads) cell of results/threads.json on 4 cores.
    # results/load_workers2_intra2.json keeps the first run with 2 x 2 for comparison.
    configs = {"batching_off": dict(max_batch=1, wait_us=0, workers=4, intra=1),
               "batching_on": dict(max_batch=16, wait_us=4000, workers=4, intra=1)}
    for model, kind in (("resnet50_v2_int8", "vision"), ("distilbert_sst2_int8", "text")):
        start_server(model, **configs["batching_off"])
        probe = loadgen(400, 6, kind)
        cap = max(1.0, probe["goodput_rps"])
        print(f"  {model}: batching-off capacity ~{cap:.1f} rps")
        entry = {"capacity_rps_batching_off": round(cap, 1), "points": []}
        for name, cfg in configs.items():
            start_server(model, **cfg)
            loadgen(cap * 0.5, 3, kind)  # warm-up
            for mult in (0.5, 1, 2, 4):
                d = loadgen(cap * mult, seconds, kind)
                d.update({"config": name, "load_x_capacity": mult, **cfg})
                entry["points"].append(d)
                print(f"    {name:13s} {mult:>3}x  goodput {d['goodput_rps']:7.1f}  p50 {d['p50_ms']:7.1f}  p95 {d['p95_ms']:8.1f}  "
                      f"503 {d['rejected_503']:5d}  batch {d['mean_batch_size']:.2f}")
        out["models"][model] = entry
        subprocess.run(["docker", "rm", "-f", "edge-sut"], capture_output=True)
    (RES / "load.json").write_text(json.dumps(out, indent=2))


def micro() -> None:
    drun(["./build/release/bench/edgeinfer_bench", "--benchmark_format=json", "--benchmark_out=/src/results/microbench.json",
          "--benchmark_repetitions=3", "--benchmark_report_aggregates_only=true"], capture=False)


def parity() -> None:
    for model, kind in (("resnet50_v2_fp32", "vision"), ("resnet50_v2_int8", "vision"), ("distilbert_sst2_fp32", "text"),
                        ("distilbert_sst2_int8", "text")):
        start_server(model, max_batch=8, wait_us=2000, workers=1, intra=2)
        extra = ["--images-dir", "models/parity_images", "--n", "32"] if kind == "vision" else []
        drun(["python3", "tools/parity.py", "--url", "http://edge-sut:8080", "--model", f"models/{model}/model.onnx", "--kind", kind,
              "--out", f"/src/results/parity_{model}.json", *extra], cpus=CLIENT_CPUS, capture=False)
    subprocess.run(["docker", "rm", "-f", "edge-sut"], capture_output=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="env,latency,threads,load,micro,parity")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-build", action="store_true")
    a = ap.parse_args()
    RES.mkdir(exist_ok=True)
    subprocess.run(["docker", "network", "create", NET], capture_output=True)
    if not a.no_build:
        build()
    steps = a.only.split(",")
    for step in steps:
        print(f"== {step}", flush=True)
        t0 = time.time()
        {"env": env, "latency": lambda: latency(a.quick), "threads": lambda: threads(a.quick), "load": lambda: load(a.quick),
         "micro": micro, "parity": parity}[step]()
        print(f"   {step} done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
