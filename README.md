# swiftperm-bench

Benchmarking permutation inference on Apple silicon — comparing Python (NumPy, Numba) against Swift with CPU (Accelerate) and GPU (Metal) backends.

## Motivation

Permutation tests are the workhorse of nonparametric spatial statistics. They're embarrassingly parallel, computationally expensive at scale, and currently underserved by Apple-optimized implementations in mainstream scientific software (R, Python/SciPy, libpysal).

This project asks: **how fast can we run permutation inference on Apple silicon, and what does that unlock for spatial research?**

The test statistic is **Moran's I** — a standard measure of spatial autocorrelation used across epidemiology, ecology, econometrics, and geography. The framework is designed to generalize to arbitrary permutation statistics.

## Results

Benchmarked on **Apple M3, 8GB unified memory, 8 cores**, 99,999 permutations.

### Full comparison (perm/s, higher is better)

| Implementation | n=49 | n=500 | n=2,000 | n=5,000 | n=10,000 |
|---|---|---|---|---|---|
| NumPy (baseline) | 191k | 77k | 25k | 11k | 5.7k |
| Numba parallel | 5.4M | 173k | 66k | 31k | 17k |
| Swift parallel (8t) | 3.1M | **349k** | **88k** | **34k** | **17k** |
| Swift+Metal | **14.85M** | **294k** | 44k | 13k | 5.4k† |

† 2-batch fallback — index buffer exceeds 40% of `recommendedMaxWorkingSetSize` on 8GB M3. CPU parallel wins at n≥2,000.

### Speedup over NumPy

| Implementation | n=49 | n=500 | n=2,000 | n=5,000 | n=10,000 |
|---|---|---|---|---|---|
| Numba parallel | 28.3x | 2.2x | 2.7x | 2.9x | 2.9x |
| Swift parallel (8t) | 16.2x | **4.5x** | **3.5x** | **3.1x** | **2.9x** |
| Swift+Metal | **77.8x** | **3.8x** | 1.8x | 1.2x | 0.9x† |

### Metal shader variants

The Metal implementation automatically selects the best shader for each dataset size, querying `device.recommendedMaxWorkingSetSize` at runtime to adapt to available hardware:

| Condition | Shader | Strategy | Device memory |
|---|---|---|---|
| n ≤ 256 | `moranPermutationScratchless` | uint16 perm on thread stack | ~0 |
| index buf ≤ 40% RAM | `moranPermutationIndexed` | uint32 index buffer, single pass | n × nPerm × 4B |
| otherwise | `moranPermutation` | float scratch, batched | budget/batch |

On 8GB M3, this means n=49–5,000 all run as single-pass GPU dispatches. Only n=10,000 falls to 2-batch.

### Key findings

**Metal peak: 14.85M perm/s (77.8x over NumPy)** at n=49 using the stack-allocated scratchless shader on M3. Zero device memory allocation beyond the output buffer. This is the M3 GPU running flat out with unified memory — no CPU↔GPU transfer cost whatsoever.

**Runtime memory budget extends single-pass Metal to n=5,000.** Using 40% of `device.recommendedMaxWorkingSetSize` as the indexed buffer budget allows n=2,000 and n=5,000 to run as single-pass indexed dispatches on 8GB M3, automatically scaling to larger machines without code changes.

**Swift parallel is the best general-purpose CPU implementation.** At n ≥ 500, Swift's ahead-of-time compiled parallel loop consistently beats both Numba (4.5x vs 2.2x over NumPy at n=500) and Metal for large n. No JIT warmup, no Python runtime, near-linear parallel scaling across all 8 cores.

**Metal crossover: n ≈ 2,000 on 8GB M3.** Single-pass Metal wins at n ≤ 500 comfortably, and remains competitive to n=5,000 (1.2x over NumPy). CPU parallel takes over from n=2,000 where Swift parallel's 3.5x advantage over NumPy exceeds Metal's 1.8x.

**Three-tier shader architecture eliminates the fixed-scratch bottleneck.** Previous batched float-scratch implementation was limited to small n or many sequential dispatches. The indexed uint32 buffer approach is 4x smaller than float scratch and runs in one pass for most practical research dataset sizes.

### Parallel CPU scaling

Swift's `DispatchQueue.concurrentPerform` achieves 5–8.5x speedup over serial on 8 cores — near-linear, confirming the permutation loop is embarrassingly parallel with negligible synchronization overhead.

### Why Swift parallel beats Numba at scale

At n ≥ 500, Swift's ahead-of-time LLVM compilation generates more aggressively vectorized shuffle and dot-product code than Numba's JIT, with no runtime warmup cost. Consistent 3–4.5x over NumPy across all sizes above n=500, vs Numba's 2.2–2.9x.

## Implications for research

Currently, published spatial analysis studies routinely cap permutation counts at 999 or 9,999 because 99,999 permutations is too slow. With Swift+Metal on M3 at n=49, or Swift parallel at n=5,000:

| Permutations | Metal (n=49) | Swift parallel (n=5k) |
|---|---|---|
| 999 | < 0.1ms | 32ms |
| 9,999 | 0.7ms | 0.3s |
| 99,999 | 6.7ms | 3.2s |
| 999,999 | ~67ms | ~32s |

A million permutations in under 100ms at n=49. A million permutations in 32 seconds at n=5,000. Neither is feasible in standard Python workflows today.

## Structure

```
swiftperm-bench/
├── SwiftGeo/                   # Swift package
│   └── Sources/
│       ├── SwiftGeo/
│       │   ├── MoranPermutation.swift      # CPU: serial + parallel (Accelerate)
│       │   ├── MoranPermutation.metal      # GPU: three shader variants
│       │   ├── MetalPermutation.swift      # GPU: Metal host + runtime shader selection
│       │   └── BinaryIO.swift             # Binary fixture I/O
│       ├── SwiftGeoCLI/
│       │   └── main.swift                 # CLI runner
│       └── SwiftGeoLib/
│           └── bridge.swift               # C bridge for Python ctypes binding
├── python/
│   ├── generate_fixtures.py               # Columbus dataset → binary
│   ├── generate_synthetic.py              # Synthetic KNN datasets
│   ├── swiftperm.py                       # Python ctypes wrapper for libSwiftGeoLib.dylib
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

# 3. Run full benchmark (Python only)
cd ../python
uv run python benchmark.py --synthetic

# 3b. Run with Swift/Metal rows (requires step 2)
uv run python benchmark.py --synthetic --dylib
```

## Python bindings

`python/swiftperm.py` provides a ctypes wrapper around `libSwiftGeoLib.dylib`, exposing all three backends directly from Python with no subprocess overhead and no file I/O.

```python
from swiftperm import SwiftPerm
import numpy as np

sp = SwiftPerm()  # auto-discovers .build/release/libSwiftGeoLib.dylib

# z: float64 array, rows/cols: int32 COO indices, vals: float64 weights
result = sp.perm_parallel(z, rows, cols, vals, n, n_perm=99999)
print(result.observed, result.p_value, result.elapsed_seconds)

# All three backends available individually:
result = sp.perm_serial(z, rows, cols, vals, n)
result = sp.perm_parallel(z, rows, cols, vals, n)
result = sp.perm_metal(z, rows, cols, vals, n)   # falls back to parallel if Metal unavailable
```

**Design:** Python pre-allocates the null distribution buffer (`np.empty(n_perm, float64)`); Swift writes directly into it. No heap allocation or ownership transfer — Python always owns the memory. The only copy is a single `memcpy` of `nPerm × 8` bytes after computation completes.

**Override dylib path:** `SWIFTGEO_DYLIB=/path/to/libSwiftGeoLib.dylib` or pass `dylib_path=` to `SwiftPerm()`.

### Subprocess vs. ctypes overhead (M3, n=49, 9,999 permutations)

| Approach | Wall time/call | vs. ctypes |
|---|---|---|
| Subprocess + file I/O (old) | 116ms | — |
| ctypes `perm_parallel` (new) | 2.5ms | **47x faster** |
| ctypes `perm_metal` (new) | 2.5ms | **46x faster** |

The subprocess fixed cost (~114ms) dominated total call time at small n — larger than the Metal computation itself (≈0.7ms at n=49 with 9,999 perms). The ctypes binding eliminates this overhead entirely, making rapid iteration across many model specifications or dataset sizes practical. At large n (where computation >> 114ms), the relative gain is smaller but the binding is never slower.

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
- [x] Metal GPU — uint32 indexed single-pass (n ≤ 40% RAM)
- [x] Runtime memory budget via `recommendedMaxWorkingSetSize`
- [x] Python binding via ctypes (`python/swiftperm.py` → `libSwiftGeoLib.dylib`)
- [ ] Larger synthetic datasets: n=50k, n=100k
- [ ] Generalize beyond Moran's I to arbitrary permutation statistics
- [ ] R package wrapper
- [ ] Benchmark on M3 Pro/Max/Ultra (larger memory budgets)

## Background

This work extends earlier comparisons of Python and GPU implementations of spatial statistics to the Apple silicon platform — specifically examining whether unified memory architecture eliminates the CPU↔GPU transfer penalty that typically limits GPU acceleration for moderate-sized statistical workloads. On M3 with a scratchless single-pass GPU dispatch, the answer is unambiguous: **77.8x over NumPy** at n=49, with the architectural constraint being sparse matmul throughput rather than memory bandwidth for larger datasets.

## Author

Philip Stephens — [@pastephens](https://github.com/pastephens)

## License

BSD 3-Clause
