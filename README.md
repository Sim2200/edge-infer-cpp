# edge-infer-cpp

[![CI](https://github.com/Sim2200/edge-infer-cpp/actions/workflows/ci.yml/badge.svg)](https://github.com/Sim2200/edge-infer-cpp/actions)
![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C?logo=cplusplus&logoColor=white)
![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.20_C%2B%2B_API-005CED?logo=onnx&logoColor=white)
![SIMD](https://img.shields.io/badge/SIMD-AVX2_%C2%B7_NEON-555)

A C++20 inference server and benchmark harness for quantized vision and language models on CPU,
built and measured on Linux. The serving layer is written by hand: a bounded MPMC queue (mutex and
condition variables) and a lock-free ring buffer to compare it with, a thread pool, dynamic
micro-batching, an HTTP endpoint, Prometheus metrics, and an AVX2/NEON preprocessing kernel. ONNX
Runtime's C++ API runs the models: ResNet-50 and MobileNetV3 (fp32 and int8) and DistilBERT
(fp32 and dynamic int8).

Every number below comes from a file in `results/`, produced by `scripts/run_experiments.py`.
`docs/architecture.md` has the design and its trade-offs; `docs/writeup.md` tells the story of the
biggest performance finding.

## Setup the numbers come from (`results/env.json`)

| | |
|---|---|
| CPU | Intel Core i7-9750H @ 2.60 GHz (Coffee Lake, AVX2 + FMA) |
| System under test | Docker container pinned to **4 cores** (`--cpuset-cpus 0-3`) |
| Load generator | Separate container pinned to 4 other cores (`6-9`), so the client never competes with the server |
| OS / compiler | Ubuntu 24.04.5 LTS, GCC 13.3.0, `-O3 -DNDEBUG`, `-mavx2 -mfma` on the AVX2 translation unit only |
| Runtime | ONNX Runtime 1.20.1 (official prebuilt, CPU execution provider) |
| Host | Docker Desktop VM on an Intel MacBook Pro (12 logical CPUs, 8 GiB to the VM) |
| ARM | GitHub Actions `ubuntu-24.04-arm` runner, 4 vCPUs (preprocessing benchmark only) |

## Results

### 1. fp32 vs int8, per model (`results/latency.json`)

Direct calls through the C++ wrapper (no HTTP), 4 intra-op threads, 40 timed calls after 5 warm-ups.
Peak RSS is the whole process after loading the model and running both batch sizes.

| Model | File | Load | Batch 1 p50 / p95 | Batch 8 p50 | Batch 8 items/s | Peak RSS |
|---|---|---|---|---|---|---|
| MobileNetV3-L fp32 | 20.9 MiB | 246 ms | 15.70 / 29.60 ms | 61.57 ms | 129.9 | 151.7 MiB |
| MobileNetV3-L int8 | 5.7 MiB | 135 ms | **10.10** / 12.88 ms | 41.32 ms | 193.6 | **57.9 MiB** |
| ResNet-50 v2 fp32 | 97.4 MiB | 600 ms | 47.99 / 80.88 ms | 262.00 ms | 30.5 | 292.4 MiB |
| ResNet-50 v2 int8 | 24.9 MiB | 198 ms | **23.35** / 27.91 ms | 136.95 ms | 58.4 | **92.7 MiB** |
| DistilBERT SST-2 fp32 (128 tokens) | 255.4 MiB | 1,053 ms | 57.39 / 112.16 ms | 389.13 ms | 20.6 | 472.2 MiB |
| DistilBERT SST-2 int8 (128 tokens) | 64.1 MiB | 446 ms | **50.14** / 58.23 ms | 388.58 ms | 20.6 | **177.6 MiB** |

int8 halves ResNet-50 latency (2.06x) and cuts its peak memory by 68%; MobileNetV3 gets 1.55x and
62%. DistilBERT's int8 is *dynamic* quantization (weights int8, activations quantized on the fly at
run time), so it saves 62% of memory but only 13% of single-request latency, and nothing at batch 8.
For an edge device with little RAM, int8 is the difference between fitting the transformer or not;
for latency it matters much more on the CNNs.

### 2. C++ server vs Python ONNX Runtime on the same inputs (`results/parity_*.json`)

`tools/parity.py` sends real ImageNetV2 photos (raw RGB) and fixed sentences to the C++ server and
compares its logits with Python ONNX Runtime fed by an independent NumPy implementation of the same
preprocessing.

| Model | Inputs | Same top-1 | Max abs logit difference |
|---|---|---|---|
| ResNet-50 v2 fp32 | 32 photos | 32 / 32 | 6.4e-5 |
| ResNet-50 v2 int8 | 32 photos | 31 / 32 | 1.06 |
| DistilBERT fp32 | 8 sentences | 8 / 8 | 0.0 |
| DistilBERT int8 | 8 sentences | 8 / 8 | 0.0 |

The fp32 row is the correctness proof for the C++ preprocessing (float32 SIMD vs float64 NumPy, 6.4e-5
apart). The int8 row is a property of quantized models, not a bug: a pixel value one float rounding
step away can fall into a neighbouring int8 bin, and quantization amplifies it, so one near-tie photo
flips. Text inputs are identical token ids, hence exactly zero. Building this check also caught a bug
in the *reference*: its first version subtracted `uint8` pixels, which wrap around below zero, so it
scored far below the C++ server against the photos' true labels until the values were cast to float.

### 3. Preprocessing kernel: SIMD vs scalar (`results/microbench.json`, CI artifact)

RGB8 -> bilinear resize to 224x224 -> normalise to float CHW. Google Benchmark, medians of 3
repetitions. x86-64 on one of the pinned cores; ARM on the GitHub arm64 runner.

| Input | x86 scalar | x86 **AVX2** | Speedup | ARM scalar | ARM NEON | Speedup |
|---|---|---|---|---|---|---|
| 640x480 | 495.6 us | 264.5 us | **1.87x** | 345.0 us | 322.6 us | 1.07x |
| 1920x1080 | 479.1 us | 274.6 us | **1.74x** | 353.2 us | 336.6 us | 1.05x |

The AVX2 path issues one 32-bit `vpgatherdd` per bilinear corner for 8 output pixels and pulls the
R, G and B bytes out with per-lane variable shifts; it matches the scalar reference to 1e-5 on every
tested size, including the row-end and tail cases. My first AVX2 version loaded the bytes one at a time and
was much slower than this one. **NEON has no gather instruction**, so the NEON path still collects bytes
with scalar loads and only computes in vector registers, and it barely beats scalar. The right ARM
design de-interleaves rows with `vld3` and does separable passes; it is future work, not claimed here.

### 4. Mutex queue vs lock-free ring (`results/microbench.json`)

1,000,000 ints through a capacity-1024 queue, P producers and P consumers, on the 4 pinned cores.

| P producers + P consumers | BoundedQueue (mutex + 2 CVs) | MpmcRing (lock-free) | Ring / mutex |
|---|---|---|---|
| 1 + 1 | 4.84 M items/s | 49.00 M items/s | 10.1x |
| 2 + 2 | 2.80 M items/s | 11.67 M items/s | 4.2x |
| 4 + 4 (8 threads on 4 cores) | 2.86 M items/s | 8.92 M items/s | 3.1x |

The ring wins at every level, by the most when contention is lowest. The server still uses the
mutex queue, deliberately: it carries one request per millisecond-scale inference, so 2.8 M ops/s is
three orders of magnitude more than needed, and it can *sleep* (the ring's waiting consumers must spin
or yield, which costs the inference threads CPU on a 4-core box).

### 5. Threads: oversubscription and spinning (`results/threads.json`)

Concurrent callers x ONNX Runtime intra-op threads on one shared session, batch 1, 4 pinned cores.
ResNet-50 int8, selected cells:

| Callers x intra-op threads | Threads | Inferences/s | p95 |
|---|---|---|---|
| 1 x 4 | 4 | 36.9 | 43.8 ms |
| **4 x 1** | 4 | **73.3** | 67.7 ms |
| 8 x 8 (oversubscribed) | 64 | 48.7 | 256.5 ms |
| 4 x 8 with `allow_spinning` | 32 | **13.9** | **722.9 ms** |
| 4 x 8 without spinning | 32 | 58.3 | 90.4 ms |

Four single-threaded sessions run twice as fast as one four-threaded session on the same cores.
Oversubscription costs throughput and multiplies tail latency, and ORT's spin-waiting threads make it
dramatically worse (4.2x lower throughput at 32 threads) by burning the CPU the other threads need. For
MobileNetV3 with exactly one caller, spinning helped (218.3 vs 102.7 inferences/s at 4 threads), which
is why it is a measured choice and not a rule. The server leaves spinning off.

### 6. The server under open-loop load (`results/load.json`, `results/load_workers2_intra2.json`)

`edge_loadgen` schedules Poisson arrivals and measures latency from the scheduled time (no coordinated
omission). Capacity is measured first (a 6 s overload probe with batching off), and load levels are
multiples of it. 20 s per point.

**The thread sweep changed the server's configuration.** The first load run used 2 batcher workers x 2
intra-op threads. Section 5 says 4 x 1 is the better cell, and the rerun confirms it:

| Server config (4 cores) | ResNet-50 int8 capacity | DistilBERT int8 capacity |
|---|---|---|
| 2 workers x 2 intra-op threads | 44.9 req/s | 24.5 req/s |
| **4 workers x 1 intra-op thread** | **74.4 req/s (+66%)** | 26.8 req/s |

With 4 x 1, batching off vs on (`max_batch` 16, `max_wait` 4 ms):

| Model | Load | Batching off: goodput, p50 / p95 | Batching on: goodput, p50 / p95, mean batch |
|---|---|---|---|
| ResNet-50 int8 | 0.5x | 37.2 req/s, 58.9 / 88.7 ms | 37.2 req/s, 66.1 / 143.5 ms, 1.21 |
| ResNet-50 int8 | 1x | 63.8 req/s, 1,318 / 3,104 ms | 60.6 req/s, 2,464 / 4,639 ms, 15.09 |
| ResNet-50 int8 | 2x | 57.5 req/s, 17.0 / 30.4 s | 59.2 req/s, 15.7 / 29.1 s, 15.87 |
| DistilBERT int8 | 0.5x | 13.8 req/s, 99.0 / 130.0 ms | 13.8 req/s, 107.0 / 168.5 ms, 1.04 |
| DistilBERT int8 | 1x | 25.4 req/s, 145.8 / 305.3 ms | 25.4 req/s, 117.9 / 529.6 ms, 1.64 |
| DistilBERT int8 | 2x | 24.3 req/s, 7.6 / 22.7 s | **30.9 req/s**, 7.2 / **14.4 s**, 15.36 |

Honest readings:

- **Batching does not raise CNN throughput on CPU.** ResNet-50 is compute-bound: a batch of 16 costs
  about 16 single images, so batching only adds waiting (p95 at half load 88.7 -> 143.5 ms). At 4x the
  batching run was lower still (50.4 vs 59.5 req/s).
- **For DistilBERT it helped under overload**: at 2x, +27% goodput and 37% lower p95. At 4x the
  batching-off run was higher (31.4 vs 27.9 req/s), so with one run per point I call 4x inconclusive.
- **No 503s anywhere**: the queue holds 256 requests and the load generator never has more than 128 in
  flight, so overload turns into seconds of queueing instead of fast rejections. The queue bound should
  be derived from the latency target, not memory; see `docs/architecture.md`.

All 4x rows, the 2 x 2 run and the batching-on 0.5x/1x details for both models are in the JSON files.

## Engineering

- **Build**: CMake 3.24+ with presets (`release`, `debug`, `asan`), all dependencies through
  `FetchContent` pinned to releases (GoogleTest 1.15.2, Google Benchmark 1.9.1, cpp-httplib 0.18.3,
  nlohmann/json 3.11.3, ONNX Runtime 1.20.1 prebuilt for x64 and aarch64). Builds warning-free under
  `-Wall -Wextra -Wpedantic -Wshadow -Wconversion`.
- **Tests**: 38 GoogleTest cases: queue and ring (including 4x4 producer/consumer stress that checks
  every item arrives exactly once), thread pool, preprocessing (SIMD vs scalar to 1e-5 at awkward
  sizes, row-end clamps, padded strides), batcher (batch limits, rejection, error propagation, output
  size validation), metrics, and the model wrapper on real models.
- **Sanitizers**: the whole suite runs under AddressSanitizer + UndefinedBehaviorSanitizer. It found a
  real heap overflow (the batcher trusted the model's output size), fixed and covered by a test.
  Leak reports from ONNX Runtime's process-lifetime allocations are suppressed by library name only
  (`lsan.supp`); nothing of ours is suppressed.
- **CI** (`.github/workflows/ci.yml`): Release build + tests and ASan/UBSan build + tests on Ubuntu
  24.04 x86-64, and a native arm64 build + tests + NEON benchmark on `ubuntu-24.04-arm`.

## Running it

```bash
docker build -t edge-infer-dev -f docker/Dockerfile docker
tools/fetch_models.sh                      # copies/exports the six ONNX models into models/ (see the script)
docker run --rm -v $PWD:/src -v edge-infer-main:/src/build edge-infer-dev \
  bash -c 'cmake --preset release && cmake --build --preset release && ctest --preset release'
python3 scripts/run_experiments.py         # every results/*.json (about 35 minutes)

# serve one model
docker run --rm -p 8080:8080 -v $PWD:/src -v edge-infer-main:/src/build edge-infer-dev \
  ./build/release/edge_server --model models/resnet50_v2_int8/model.onnx --workers 4 --intra 1 --max-batch 16
curl -s localhost:8080/health ; curl -s localhost:8080/metrics | head
```

## Layout

```
include/edgeinfer/   bounded_queue.hpp · mpmc_ring.hpp · thread_pool.hpp · batcher.hpp · model.hpp · preprocess.hpp · metrics.hpp
src/                 batcher · model (ONNX Runtime) · preprocess_{scalar,avx2,neon,dispatch} · metrics · server_main · bench_cli · loadgen
tests/               GoogleTest suites          bench/   Google Benchmark (kernel, queues)
tools/               parity.py · fetch_models.sh          scripts/   run_experiments.py
docs/                architecture.md · writeup.md         results/   every number above
```

## Limitations

- One machine class (a 2019 laptop CPU in a VM, 4 pinned cores); no phone or Android NDK run. The ARM
  numbers are from a cloud runner and cover only the preprocessing kernel.
- One run per load point, 20 s each; overload points have wide variance.
- DistilBERT is fed token ids (tokenization stays on the client); images arrive as raw RGB, not JPEG.
- The vision models are fed a direct resize to 224x224 rather than resize-and-centre-crop, which
  costs some accuracy; it does not affect any comparison here because both sides of each comparison
  use the same preprocessing.
- No GPU path was measured.

## Author

**Simran Kharbanda** · [github.com/Sim2200](https://github.com/Sim2200)
