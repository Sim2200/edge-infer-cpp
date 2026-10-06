# The thread count was worth more than the batcher

*What I measured while building a C++ inference server for a 4-core CPU, and why the biggest win
was a configuration change. Every number is from `results/` in this repository.*

I built the serving path the textbook way: a bounded request queue, a pool of batcher workers, each
calling an ONNX Runtime session, with dynamic micro-batching on top. My first configuration looked
reasonable: 2 workers, each running inference with 2 intra-op threads, so 4 threads for 4 cores. Under
an open-loop load generator, ResNet-50 int8 topped out at **44.9 requests/s**.

The obvious next lever was batching. It is the standard answer to "serve more requests per second",
and the batcher was already there. So I measured it, and on this CPU it did nothing for the CNN: with
batches of 16 the server sustained about the same rate as without, and at half load batching only
added waiting time (p95 88.7 ms without, 143.5 ms with). A convolutional network on a CPU is
compute-bound. Sixteen images cost about sixteen times one image. Batching pays where per-call overhead
or memory bandwidth dominates, which is a GPU story more than a 4-core CPU one.

## The measurement that mattered

Separately, I swept how the 4 cores are split: C concurrent callers on one shared session, each call
using T intra-op threads, for every C and T in {1, 2, 4, 8}. For ResNet-50 int8:

- 1 caller x 4 threads: **36.9** inferences/s
- 4 callers x 1 thread: **73.3** inferences/s

Same 4 cores, twice the throughput. Splitting one convolution across 4 threads means synchronising at
every operator boundary, and at batch 1 the operators are small enough that the synchronisation eats
much of the gain. Four independent single-threaded inferences never wait for each other.

The same sweep showed what goes wrong in the other direction. At 32 threads on 4 cores, throughput was
58.3/s with ONNX Runtime's spin-waiting off and **13.9/s with it on**, with p95 at 723 ms. Spinning
keeps idle worker threads busy-waiting for the next operator; when the cores are oversubscribed, those
spinning threads steal exactly the CPU the working threads need. I had seen the same effect in a
Python version of this gateway as CFS throttling inside a CPU-limited container, so the server ships
with spinning off.

## Closing the loop

I changed one line of server configuration to the best cell of the sweep, 4 batcher workers x 1
intra-op thread, and re-ran the load test. ResNet-50 int8 capacity went from 44.9 to **74.4 requests/s,
+66%**, with no code change. For DistilBERT the same change moved capacity from 24.5 to 26.8 requests/s,
and there batching did help under overload (+27% goodput and 37% lower p95 at twice capacity), because
a transformer at batch 1 leaves more of the CPU's vector units idle than a CNN does.

## What I take from it

- Measure the knob you are about to turn before you build the feature around it. Batching was the
  feature I expected to matter; the thread split was the one that did.
- Report the results that did not help. Batching for the CNN, the lock-free queue (10x faster than the
  mutex queue at one producer, irrelevant at a few hundred requests per second), and the NEON kernel
  (barely faster than scalar without a gather instruction) are all in the README with their numbers.
- Keep the measurement honest: an open-loop load generator that times from the scheduled send, server
  and client on separate pinned cores, and a correctness check (the C++ server against Python ONNX
  Runtime on real photos) before any speed claim.
