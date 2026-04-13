# swiftperm-bench

Benchmarking permutation inference on Apple silicon — comparing Python (NumPy, Numba) against Swift with CPU (Accelerate) and GPU (Metal) backends.

## Motivation

Permutation tests are the workhorse of nonparametric spatial statistics. They're embarrassingly parallel, computationally expensive at scale, and currently underserved by Apple-optimized implementations in mainstream scientific software (R, Python/SciPy, libpysal).

This project asks: **how fast can we run permutation inference on Apple silicon, and what does that unlock for spatial research?**

The test statistic is **Moran's I** — a standard measure of spatial autocorrelation used across epidemiology, ecology, econometrics, and geography. The framework is designed to generalize to arbitrary permutation statistics.

## Results

Benchmarked on **Apple M3, 8 cores**, 99,999 permutations.

### Full comparison (perm/s, higher is better)

| Implementation | n=49 | n=500 | n=2,000 | n=5,000 | n=10,000 |
|---|---|---|---|---|---|
| NumPy (baseline) | 193k | 78k | 25k | 11k | 5.7k |
| Numba parallel | 5.5M | 167k | 68k | 31k | 16k |
| Swift parallel (8t) | 3.1M | **349k** | **88k** | **34k** | **17k** |
| Swift+Metal (scratchless) | **11.66M** | **323k** | 40k† | 11k† | 4.6k† |

† Batched fallback (index buffer exceeds 512MB). CPU parallel wins here.

### Speedup over NumPy

| Implementation | n=49 | n=500 | n=2,000 | n=5,000 | n=10,000 |
|---|---|---|---|---|---|
| Numba parallel | 28.7x | 2.1x | 2.7x | 2.9x | 2.8x |
| Swift parallel (8t) | 16.1x | **4.5x** | **3.5x** | **3.1x** | **3.0x** |
| Swift+Metal | **60.3x** | **4.1x** | 1.6x† | 1.0x† | 0.8x† |

### Metal shader variants

The Metal implementation automatically selects the best shader for each dataset size:

| n | Shader | Strategy | Memory |
|---|---|---|---|
| ≤ 256 | `moranPermutationScratchless` | uint16 perm on thread stack | ~0 device memory |
| ≤ ~130k* | `moranPermutationIndexed` | uint32 index buffer, single pass | n × nPerm × 4B |
| > ~130k* | `moranPermutation` | float scratch, batched | 512MB/batch |

*Threshold where index buffer fits in 512MB at 99,999 permutations. Raising the limit to 1GB would bring n=2,000 into the single-pass indexed path.

### Key findings

**Metal peak: 11.66M perm/s (60.3x over NumPy)** at n=49 using the stack-allocated scratchless shader on M3. Zero device memory allocation beyond the output buffer.

**Scratchless shader: 11.3x improvement over batched Metal at n=49.** Moving the permutation index array from device memory to GPU thread stack eliminates all scratch buffer allocation and dispatch overhead for n≤256.

**Swift parallel is the best general-purpose CPU implementation.** At n ≥ 500, Swift's ahead-of-time compiled parallel loop consistently beats Numba (2.1x vs 4.5x over NumPy at n=500) and exceeds batched Metal for n ≥ 2,000. No JIT warmup, no Python runtime.

**Metal crossover point: n ≈ 500.** Single-pass Metal (scratchless or indexed) wins at n ≤ 500. CPU parallel wins from n ≥ 2,000. The gap is batch dispatch overhead (~10ms/batch) — a fixed cost that dominates as n grows.

**The indexed shader halved the crossover threshold.** Previous batched Metal (float scratch) crossed over at n=500. The uint32 indexed shader extends Metal's advantage to n=500 in a single pass, with ~1.27x improvement over batched there.

### Parallel CPU scaling

Swift's `DispatchQueue.concurrentPerform` achieves 5–8.5x speedup over serial on 8 cores — near-linear scaling confirming the permutation loop is embarrassingly parallel with negligible synchronization overhead.

### Why Swift parallel beats Numba at scale

At n ≥ 500, Swift's ahead-of-time LLVM compilation generates more aggressively vectorized shuffle and dot-product code than Numba's JIT, with no runtime warmup cost. Consistent 3–4.5x over NumPy vs Numba's 1.7–2.9x across all sizes tested above n=500.

## Implications for research

Currently, published spatial analysis studies routinely cap permutation counts at 999 or 9,999 because 99,999 permutations is too slow. With Swift+Metal on M3 at n=49, or Swift parallel at n=5,000:

| Permutations | Metal (n=49) | Swift parallel (n=5k) |
|---|---|---|
| 999 | < 0.1ms | 29ms |
| 9,999 | 0.9ms | 0.3s |
| 99,999 | 8.6ms | 3.2s |
| 999,999 | ~86ms | ~32s |

A million permutations in under a second at n=49. A million permutations in 32 seconds at n=5,000. Neither is currently feasible in standard Python workflows.

## Structure

```
swiftperm-bench/
├── SwiftGeo/                   # Swift package
│   └── Sources/
│       ├── SwiftGeo/
│       │   ├── MoranPermutation.swift      # CPU: serial + parallel (Accelerate)
│       │   ├── MoranPermutation.metal      # GPU: three shader variants
│       │   ├── MetalPermutation.swift      # GPU: Metal host + shader selection
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

## Roadmap

- [x] CPU serial implementation (Swift + Accelerate)
- [x] CPU parallel implementation (`DispatchQueue.concurrentPerform`)
- [x] Metal GPU — batched float scratch
- [x] Metal GPU — scratchless stack shader (n ≤ 256)
- [x] Metal GPU — uint32 indexed single-pass (n ≤ ~130k)
- [ ] Raise indexed limit to 1GB (captures n=2,000 as single pass)
- [ ] Larger synthetic datasets: n=50k, n=100k
- [ ] Python binding via ctypes or Swift-Python bridge
- [ ] Generalize beyond Moran's I to arbitrary permutation statistics
- [ ] R package wrapper

## Background

This work extends earlier comparisons of Python and GPU implementations of spatial statistics to the Apple silicon platform — specifically examining whether unified memory architecture eliminates the CPU↔GPU transfer penalty that typically limits GPU acceleration for moderate-sized statistical workloads. On M3 with a scratchless single-pass GPU dispatch, the answer is unambiguous: **60.3x over NumPy** at n=49, with the architectural constraint being batch dispatch overhead rather than compute throughput for larger datasets.

## Author

Philip Stephens — [@pastephens](https://github.com/pastephens)

## License

BSD 3-Clause
