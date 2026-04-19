# SwiftGeoLib

A Swift spatial statistics library optimized for Apple silicon, with a Python ctypes binding for use alongside libpysal.

## Overview

SwiftGeoLib provides CPU and GPU (Metal) implementations of global and local spatial statistics. It is designed to be called from Python as a fast backend — Python owns geometry, I/O, and weight construction (via libpysal); Swift owns computation.

The library targets the same statistical outputs as [esda](https://github.com/pysal/esda) — Moran's I, Geary's C, Getis-Ord G, join count, and their local (LISA) variants — with the goal of accelerating workflows that already use libpysal weight matrices.

## Statistics

The goal is to match [esda](https://github.com/pysal/esda) outputs exactly while using Swift Accelerate (CPU) and Metal (GPU) to outperform esda's backends on Apple silicon. `baseline.py` captures esda timings once; `benchmark.py` reports speedups against them.

**The esda baseline is already Numba-accelerated.** esda's permutation engine (`esda/crand.py`) uses `@njit(fastmath=True)` throughout and distributes work across all cores via joblib when `n_jobs=-1`. Speedups reported here are over esda + Numba + multicore — not bare NumPy.

### Global permutation tests (scalar output)

| Statistic | esda baseline | Accelerate serial | Accelerate parallel | Metal GPU |
|---|---|---|---|---|
| Moran's I | ✓ | ✓ | ✓ | ✓ |
| Geary's C | ✓ | ✓ | ✓ | ✓ |
| Getis-Ord G | ✓ | ✓ | ✓ | — |
| Join count | ✓ | ✓ | ✓ | — |

### Local / LISA (vector output, one value per observation)

| Statistic | esda baseline | Point estimate | Accelerate serial perm | Accelerate parallel perm | Metal GPU perm |
|---|---|---|---|---|---|
| Local Moran's I | ✓ | ✓ | ✓ | ✓ | — |
| Local Geary's C | ✓ | ✓ | ✓ | ✓ | — |

### Spatial lag

| | Accelerate serial | Accelerate parallel |
|---|---|---|
| W·y | ✓ | ✓ |

## Python API

`python/swiftperm.py` wraps `libSwiftGeoLib.dylib` via ctypes. All inputs are standard NumPy arrays; output buffers are pre-allocated by Python (Swift writes directly into them — no heap transfer).

```python
from swiftperm import SwiftPerm, w_to_coo
import libpysal

w = libpysal.weights.Queen.from_shapefile(...)
rows, cols, vals = w_to_coo(w)   # libpysal W → COO int32/float64

sp = SwiftPerm()   # auto-discovers .build/release/libSwiftGeoLib.dylib

# Global permutation test — serial, parallel, or Metal GPU
result = sp.perm_parallel(z, rows, cols, vals, n, n_perm=99999)
result = sp.perm_metal(z, rows, cols, vals, n, n_perm=99999)
print(result.observed, result.p_value, result.elapsed_seconds)

# statistic= selects the test: "moran" (default), "gearysc", "getisordg", "joincount"
result = sp.perm_parallel(z, rows, cols, vals, n, statistic="gearysc")

# Local statistics — returns observed[n] + p_values[n]
result = sp.local_perm_parallel(z, rows, cols, vals, n, n_perm=9999)
result = sp.local_perm_parallel(z, rows, cols, vals, n, statistic="gearysc")
print(result.observed, result.p_values)

# Point estimates (no permutation)
sp.moran_i(z, rows, cols, vals, n)          # → float
sp.local_moran_i(z, rows, cols, vals, n)    # → np.ndarray (n,)
sp.local_gearysc(z, rows, cols, vals, n)    # → np.ndarray (n,)
sp.spatial_lag(y, rows, cols, vals, n)      # → np.ndarray (n,)
```

**Override dylib path:** `SWIFTGEO_DYLIB=/path/to/libSwiftGeoLib.dylib` or `SwiftPerm(dylib_path=...)`.

### w_to_coo

`w_to_coo(w)` is the interop bridge between libpysal and SwiftGeoLib. It extracts contiguous `int32`/`float64` COO arrays from any libpysal `W` object:

```python
from swiftperm import w_to_coo
rows, cols, vals = w_to_coo(w)  # works with Queen, Rook, KNN, kernel weights, etc.
```

## Metal GPU — shader selection

On discrete GPUs, data must cross a PCIe bus before the GPU can touch it — a fixed overhead that typically makes GPU acceleration impractical for moderate-sized statistical workloads. Apple silicon's unified memory architecture eliminates this penalty entirely: the GPU reads the same physical memory the CPU just wrote, with no copy. This makes Metal viable at dataset sizes that would be uneconomical on any non-Apple platform.

The backend automatically selects the best shader for each problem size by querying `device.recommendedMaxWorkingSetSize` at runtime:

| Condition | Shader | Strategy | Device memory |
|---|---|---|---|
| n ≤ 256 | scratchless | uint16 perm index on thread stack | ~0 |
| index buf ≤ 40% RAM | indexed | uint32 index buffer, single-pass | n × nPerm × 4B |
| otherwise | batched | float scratch buffer, sequential dispatches | budget/batch |

**Scratchless** allocates nothing on the device. Each GPU thread encodes its entire permuted index as a `uint16` array on the thread stack and sweeps the weight matrix once. Zero kernel launch overhead, zero memory pressure — the dominant path for small n.

**Indexed** pre-materialises the full `n × nPerm` permutation index as a `uint32` buffer (4 bytes per entry), then dispatches every permutation in a single kernel launch. Each GPU thread reads one permutation, computes the statistic, and writes one value to the null distribution. The single-pass structure maximises GPU utilisation. The 40% RAM threshold is conservative — it leaves headroom for the weight arrays and null distribution buffer. On 8GB M3 with 99,999 permutations this covers n ≤ ~800; at n=49 (Columbus) the index buffer is trivially small and this path runs at full GPU throughput.

**Batched** is the fallback for problems too large to index at once. The permuted z-values are written to a scratch buffer and the kernel is dispatched in sequential chunks sized to the available memory budget. Each chunk is a full GPU dispatch, so compute efficiency is preserved — but there is kernel-launch overhead per chunk and the scratch buffer write adds memory traffic. This path is memory-bandwidth-bound. On M3 base (100 GB/s) it is still faster than CPU parallel for large n; on M4 Pro (273 GB/s) or M4 Max (546 GB/s) the bandwidth advantage widens substantially.

**Scaling with hardware:** more RAM directly expands the indexed path — the single-pass threshold scales linearly with device memory. More GPU cores and higher memory bandwidth improve the batched path. The most impactful upgrade for spatial statistics workloads is a Pro or Max chip: more RAM means more of your problem fits in the fast indexed path, and the higher memory bandwidth accelerates the batched fallback for the rest.

On 8GB M3: n=49 → scratchless; n ≤ ~800 at 99,999 perms → indexed; larger n → batched.

## Structure

```
SwiftGeo/
└── Sources/
    ├── SwiftGeo/
    │   ├── Statistics.swift          # Statistic functions + spatial lag
    │   ├── LocalStatistics.swift     # Local (LISA) statistics — point estimates + perm tests
    │   ├── MoranPermutation.swift    # Generic CPU permutation infrastructure (serial + parallel)
    │   ├── MetalPermutation.swift    # GPU: Metal host, shader selection, embedded MSL shaders
    │   └── BinaryIO.swift            # Binary fixture I/O
    ├── SwiftGeoCLI/
    │   └── main.swift                # CLI runner
    └── SwiftGeoLib/
        └── bridge.swift              # C bridge for Python ctypes
python/
├── swiftperm.py                      # Python ctypes wrapper
├── test_swiftperm.py                 # Accuracy test suite (85 tests)
├── generate_fixtures.py              # Download real-world datasets → binary fixtures
├── baseline.py                       # Run native esda once → data/baselines.json
└── benchmark.py                      # Benchmark runner (loads baselines for comparison)
data/                                 # Binary fixtures (generated)
```

## Quickstart

```bash
# 1. Generate fixtures (downloads datasets from libpysal on first run)
cd python
uv run python generate_fixtures.py

# 2. Build dylib
cd ../SwiftGeo
swift build -c release

# 3. Run accuracy tests
cd ../python
uv run pytest test_swiftperm.py -v

# 4. Generate esda baselines (one-time; saves data/baselines.json)
uv run python baseline.py

# 5. Run benchmarks (loads baselines automatically if present)
uv run python benchmark.py
```

## Requirements

**To use the Python binding with a pre-built dylib:**
- macOS 13+, Apple silicon
- Python 3.11+, `uv`
- `libpysal`, `esda`, `scipy`, `numpy`

**To compile the dylib from source:**
- Xcode (required for Metal shader compilation)

## Datasets

Fixtures generated by `generate_fixtures.py` using libpysal:

| Dataset | n | Weights | Variable |
|---|---|---|---|
| Columbus neighborhood crime | 49 | Queen contiguity | crime rate |
| NCOVR US county homicides | 3,085 | Queen contiguity | HR90 |
| King County WA home sales | 21,613 | KNN-8 | log(price) |
| US SDOH census tracts | 71,901 | KNN-8 | poverty rate |
| NYC Earnings census blocks | 108,487 | KNN-8 | log1p(earnings) |

## Binary fixture format

Fixtures are raw little-endian binary for zero-overhead sharing between Python and Swift:

**z vector** (`*_z.bin`): `[int32 n][float64 × n]`

**Sparse weights** (`*_weights.bin`): `[int32 n][int32 nnz][int32 × nnz rows][int32 × nnz cols][float64 × nnz values]`

## Roadmap

- [x] CPU serial + parallel permutation infrastructure
- [x] Metal GPU — scratchless, indexed, and batched shader variants
- [x] Runtime shader selection via `recommendedMaxWorkingSetSize`
- [x] Python ctypes binding (`swiftperm.py`)
- [x] Global statistics: Moran's I, Geary's C (CPU+GPU), Getis-Ord G, join count (CPU)
- [x] Spatial lag W·y — serial and parallel
- [x] Local Moran's I (LISA) — point estimates + CPU serial/parallel permutation tests
- [x] Local Geary's C — point estimates + CPU serial/parallel permutation tests
- [x] Accuracy test suite — 85 tests across 3 datasets, all passing
- [ ] Local G / G* (Getis-Ord local statistics)
- [ ] Metal GPU paths for local statistics

## Author

Philip Stephens — [@pastephens](https://github.com/pastephens)

## License

BSD 3-Clause
