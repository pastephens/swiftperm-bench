# swiftperm-bench

Benchmarking permutation inference on Apple silicon — comparing Python (NumPy, Numba) against Swift with CPU (Accelerate) and GPU (Metal) backends.

## Motivation

Permutation tests are the workhorse of nonparametric spatial statistics. They're embarrassingly parallel, computationally expensive at scale, and currently underserved by Apple-optimized implementations in mainstream scientific software (R, Python/SciPy, libpysal).

This project asks: **how fast can we run permutation inference on Apple silicon, and what does that unlock for spatial research?**

The test statistic is **Moran's I** — a standard measure of spatial autocorrelation used across epidemiology, ecology, econometrics, and geography. The framework is designed to generalize to arbitrary permutation statistics.

## Results

Benchmarked on **Apple M3, 8 cores**, 99,999 permutations.

### Columbus dataset (n=49) — all implementations

| Implementation | perm/s | vs NumPy | vs Numba |
|---|---|---|---|
| NumPy (baseline) | 184k | 1.0x | — |
| Swift serial | 650k | 3.5x | — |
| Swift parallel (8t) | 3.06M | 16.6x | — |
| Numba parallel | 4.72M | 25.6x | — |
| **Swift + Metal (M3 GPU)** | **8.47M** | **45.9x** | **1.8x** |

### Synthetic KNN datasets — Swift parallel vs Numba

| Implementation | n=500 | n=2,000 | n=5,000 | n=10,000 |
|---|---|---|---|---|
| NumPy (baseline) | 78k | 24k | 11k | 5.7k |
| Numba parallel | 133k (1.7x) | 66k (2.7x) | 30k (2.7x) | 16k (2.8x) |
| Swift parallel (8t) | **349k (4.5x)** | **88k (3.6x)** | **34k (3.1x)** | **17k (3.0x)** |
| Swift + Metal | pending (batched dispatch) | | | |

### Key findings

**Metal GPU achieves 45.9x over NumPy and 1.8x over Numba** on the M3 at n=49. The unified memory architecture of Apple silicon means zero CPU↔GPU transfer penalty — all permutations are dispatched directly from shared memory. This is 8.47 million permutations per second on a laptop chip.

**Swift parallel beats Numba at n ≥ 500.** At n=49 (tiny, overhead-dominated), Numba's JIT-compiled loops edge out Swift parallel 25.6x vs 16.6x over NumPy. But at n=500 the picture reverses sharply: Swift achieves 4.5x over NumPy while Numba manages only 1.7x. This crossover persists across all larger sizes tested.

**Why Swift wins at scale on CPU.** Numba's parallel permutation loop amortizes JIT compilation cost poorly when n is large enough that each permutation is expensive. Swift's ahead-of-time compilation generates aggressively vectorized shuffle code with no runtime overhead. The result is consistent 3–4.5x speedup over NumPy across all sizes above n=500, vs Numba's 1.7–2.8x.

**Parallel CPU scaling is near-linear.** Swift's `DispatchQueue.concurrentPerform` achieves 5–8.5x speedup over serial on 8 cores, confirming the permutation loop is embarrassingly parallel with negligible synchronization overhead.

**Metal for larger datasets is the next frontier.** Current Metal implementation is limited to n ≤ ~500 at 99,999 permutations due to per-thread scratch buffer allocation. Batched dispatch will remove this constraint — at which point the 45.9x speedup demonstrated at n=49 is expected to extend to research-scale datasets.

## Implications for research

Currently, published spatial analysis studies routinely cap permutation counts at 999 or 9,999 because 99,999 permutations is too slow. At 8.47M perm/s on an M3 MacBook:

- 999 permutations: **0.12ms**
- 9,999 permutations: **1.2ms**
- 99,999 permutations: **12ms**
- 999,999 permutations: **118ms** ← previously impractical, now trivial

This changes what p-value precision is computationally feasible in everyday research workflows.

## Structure

```
swiftperm-bench/
├── SwiftGeo/                   # Swift package
│   └── Sources/
│       ├── SwiftGeo/
│       │   ├── MoranPermutation.swift      # CPU: serial + parallel (Accelerate)
│       │   ├── MoranPermutation.metal      # GPU: Metal compute shader
│       │   ├── MetalPermutation.swift      # GPU: Metal host code
│       │   └── BinaryIO.swift             # Binary fixture I/O
│       └── SwiftGeoCLI/
│           └── main.swift                 # CLI runner
├── python/
│   ├── generate_fixtures.py               # Columbus dataset → binary
│   ├── generate_synthetic.py              # Synthetic KNN datasets
│   └── benchmark.py                       # Python benchmarks + comparison
├── data/                                  # Binary fixtures (generated)
└── results/                               # Benchmark output JSON
```

## Requirements

- macOS 13+
- Xcode (for Metal shader compilation)
- Python 3.11+, `uv`
- libpysal, esda, numba, scipy, geopandas

## Quickstart

```bash
# 1. Generate fixtures
cd python
uv init && uv add numpy pandas geopandas libpysal esda scipy numba
uv run python generate_fixtures.py
uv run python generate_synthetic.py

# 2. Build Swift
cd ../SwiftGeo
swift build -c release

# 3. Run full benchmark
cd ../python
uv run python benchmark.py \
  --synthetic \
  --swift-bin ../SwiftGeo/.build/release/SwiftGeoCLI
```

## Binary fixture format

Fixtures are raw little-endian binary, allowing zero-overhead sharing between Python and Swift:

**z vector** (`*_z.bin`): `[int32 n][float64 × n]`

**Sparse weights** (`*_weights.bin`): `[int32 n][int32 nnz][int32 × nnz rows][int32 × nnz cols][float64 × nnz values]`

**Results** (`*.bin`): `[int32 nPerm][float64 × nPerm null][float64 observed][float64 pValue][float64 elapsed][int32 nThreads]`

## Metal scratch buffer constraint

The current Metal implementation allocates a per-thread scratch buffer of size `nPerm × n × 4 bytes`. This limits single-pass Metal dispatch to:

| n | Scratch | Metal |
|---|---|---|
| 49 | 19MB | ✓ |
| 500 | 191MB | ✓ |
| 2,000 | 763MB | ✗ |
| 5,000 | 1,907MB | ✗ |

Batched Metal dispatch (chunking permutations) is the next implementation target.

## Background

This work extends earlier comparisons of Python and GPU implementations of spatial statistics to the Apple silicon platform — specifically examining whether unified memory architecture eliminates the CPU↔GPU transfer penalty that typically limits GPU acceleration for moderate-sized statistical workloads. On M3, the answer is unambiguous: Metal with unified memory achieves 45.9x over NumPy with no data transfer overhead.

## Author

Philip Stephens — [@pastephens](https://github.com/pastephens)

## License

BSD 3-Clause
