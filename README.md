# swiftperm-bench

Benchmarking permutation inference on Apple silicon — comparing Python (NumPy, Numba) against Swift with CPU (Accelerate) and GPU (Metal) backends.

## Motivation

Permutation tests are the workhorse of nonparametric spatial statistics. They're embarrassingly parallel, computationally expensive at scale, and currently underserved by Apple-optimized implementations in mainstream scientific software (R, Python/SciPy, libpysal).

This project asks: **how fast can we run permutation inference on Apple silicon, and what does that unlock for spatial research?**

The primary test statistic is **Moran's I** — a standard measure of spatial autocorrelation used across epidemiology, ecology, econometrics, and geography. The framework supports a fixed set of permutation statistics: Moran's I, Geary's C, Getis-Ord G, and join count. Geary's C also has a Metal GPU path.

## Results

Benchmarked on **Apple M3, 8GB unified memory, 8 cores**, 99,999 permutations.

Datasets are real-world spatial datasets from libpysal: Columbus neighborhood crime (n=49), King County WA home sales (n=21,613), NCOVR US county homicides (n=3,085), US SDOH census tracts (n=71,901), and NYC Earnings census blocks (n=108,487).

*Full benchmark results pending.*

### Metal shader variants

The Metal implementation automatically selects the best shader for each dataset size, querying `device.recommendedMaxWorkingSetSize` at runtime to adapt to available hardware:

| Condition | Shader | Strategy | Device memory |
|---|---|---|---|
| n ≤ 256 | scratchless | uint16 perm on thread stack | ~0 |
| index buf ≤ 40% RAM | indexed | uint32 index buffer, single pass | n × nPerm × 4B |
| otherwise | batched | float scratch, sequential dispatches | budget/batch |

On 8GB M3: n=49 → scratchless (zero device allocation); n ≤ ~5,000 → single-pass indexed; larger n → batched.

## Structure

```
swiftperm-bench/
├── SwiftGeo/                   # Swift package
│   └── Sources/
│       ├── SwiftGeo/
│       │   ├── Statistics.swift            # Statistic functions (Moran's I, Geary's C, G, join count)
│       │   ├── MoranPermutation.swift      # Generic CPU permutation infrastructure (serial + parallel)
│       │   ├── MetalPermutation.swift      # GPU: Metal host, shader selection, embedded MSL shaders
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
# 1. Generate fixtures (downloads datasets from libpysal on first run)
cd python
uv init && uv add numpy pandas geopandas libpysal esda scipy numba
uv run python generate_fixtures.py

# 2. Build Swift
cd ../SwiftGeo
swift build -c release

# 3. Run benchmark (Python only)
cd ../python
uv run python benchmark.py

# 3b. Run with Swift/Metal rows (requires step 2)
uv run python benchmark.py --dylib
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

# statistic= selects the test: "moran" (default), "gearysc", "getisordg", "joincount"
result = sp.perm_parallel(z, rows, cols, vals, n, statistic="gearysc")

# Three backends: serial, parallel (CPU), metal (GPU, falls back to parallel if unavailable)
result = sp.perm_serial(z, rows, cols, vals, n)
result = sp.perm_parallel(z, rows, cols, vals, n)
result = sp.perm_metal(z, rows, cols, vals, n)
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
- [x] Generalize beyond Moran's I: Geary's C (CPU+GPU), Getis-Ord G, join count (CPU)
- [x] Real-world datasets: Columbus, King County WA, NCOVR, US SDOH, NYC Earnings
- [ ] Full benchmark results on real datasets
- [ ] R package wrapper
- [ ] Benchmark on M3 Pro/Max/Ultra (larger memory budgets)

## Background

This work extends earlier comparisons of Python and GPU implementations of spatial statistics to the Apple silicon platform — specifically examining whether unified memory architecture eliminates the CPU↔GPU transfer penalty that typically limits GPU acceleration for moderate-sized statistical workloads. On M3 with a scratchless single-pass GPU dispatch, the answer is unambiguous: **77.8x over NumPy** at n=49, with the architectural constraint being sparse matmul throughput rather than memory bandwidth for larger datasets.

## Author

Philip Stephens — [@pastephens](https://github.com/pastephens)

## License

BSD 3-Clause
