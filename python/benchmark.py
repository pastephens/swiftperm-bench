#!/usr/bin/env python3
"""
benchmark.py
Run Moran's I permutation test across multiple dataset sizes.
Compares NumPy, Numba (parallel), and Swift (serial, parallel, Metal).

Usage:
  uv run python benchmark.py                          # Columbus only, Python
  uv run python benchmark.py --synthetic              # Add synthetic datasets
  uv run python benchmark.py --synthetic --dylib      # Include Swift/Metal rows
  uv run python benchmark.py --n-perm 9999            # Fewer permutations for quick runs
"""

import argparse
import json
import struct
import time
import subprocess
from pathlib import Path

import numpy as np

DATA_DIR    = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
N_PERM      = 99999
SEED        = 12345


# ---------------------------------------------------------------------------
# Binary I/O
# ---------------------------------------------------------------------------

def read_z_vector(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        n = struct.unpack("<i", f.read(4))[0]
        return np.frombuffer(f.read(8 * n), dtype=np.float64).copy()


def read_sparse_weights(path: Path):
    with open(path, "rb") as f:
        n   = struct.unpack("<i", f.read(4))[0]
        nnz = struct.unpack("<i", f.read(4))[0]
        rows   = np.frombuffer(f.read(4 * nnz), dtype=np.int32).copy()
        cols   = np.frombuffer(f.read(4 * nnz), dtype=np.int32).copy()
        values = np.frombuffer(f.read(8 * nnz), dtype=np.float64).copy()
    return n, nnz, rows, cols, values


def read_swift_results(path: Path):
    with open(path, "rb") as f:
        n_perm    = struct.unpack("<i", f.read(4))[0]
        null      = np.frombuffer(f.read(8 * n_perm), dtype=np.float64).copy()
        observed  = struct.unpack("<d", f.read(8))[0]
        p_value   = struct.unpack("<d", f.read(8))[0]
        elapsed   = struct.unpack("<d", f.read(8))[0]
        n_threads = struct.unpack("<i", f.read(4))[0]
    return null, observed, p_value, elapsed, n_threads


# ---------------------------------------------------------------------------
# Python implementations
# ---------------------------------------------------------------------------

def build_sparse_matrix(n, rows, cols, values):
    from scipy.sparse import csr_matrix
    return csr_matrix((values, (rows, cols)), shape=(n, n))


def moran_i_scipy(z, W):
    return float(z @ W.dot(z)) / len(z)


def moran_perm_numpy(z, W, n_perm=N_PERM, seed=SEED):
    rng = np.random.default_rng(seed)
    observed = moran_i_scipy(z, W)
    null = np.empty(n_perm)
    z_perm = z.copy()
    t0 = time.perf_counter()
    for i in range(n_perm):
        rng.shuffle(z_perm)
        null[i] = moran_i_scipy(z_perm, W)
    elapsed = time.perf_counter() - t0
    p_value = float(np.mean(np.abs(null) >= abs(observed)))
    return observed, null, p_value, elapsed


_numba_compiled = False

def moran_perm_numba(z, rows, cols, values, n, n_perm=N_PERM, seed=SEED):
    global _numba_compiled
    try:
        from numba import njit, prange
    except ImportError:
        return None

    @njit(parallel=False, cache=True)
    def _moran_i(z, rows, cols, values, n):
        wz = np.zeros(n)
        for k in range(len(values)):
            wz[rows[k]] += values[k] * z[cols[k]]
        s = 0.0
        for i in range(n):
            s += z[i] * wz[i]
        return s / n

    @njit(parallel=True, cache=True)
    def _perm_loop(z, rows, cols, values, n, n_perm, seed):
        null = np.empty(n_perm)
        for p in prange(n_perm):
            z_p = z.copy()
            state = seed + np.uint64(p) * np.uint64(6364136223846793005)
            for i in range(n - 1, 0, -1):
                state = state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
                j = int(state >> np.uint64(33)) % (i + 1)
                z_p[i], z_p[j] = z_p[j], z_p[i]
            null[p] = _moran_i(z_p, rows, cols, values, n)
        return null

    if not _numba_compiled:
        print("    Warming up Numba JIT...")
        _moran_i(z, rows, cols, values, n)
        _perm_loop(z, rows, cols, values, n, 10, np.uint64(seed))
        _numba_compiled = True

    observed = float(_moran_i(z, rows, cols, values, n))
    t0 = time.perf_counter()
    null = _perm_loop(z, rows, cols, values, n, n_perm, np.uint64(seed))
    elapsed = time.perf_counter() - t0
    p_value = float(np.mean(np.abs(null) >= abs(observed)))
    return observed, null, p_value, elapsed


# ---------------------------------------------------------------------------
# Swift binding (ctypes, direct in-process)
# ---------------------------------------------------------------------------

_swiftperm = None

def _get_swiftperm(dylib_path=None):
    global _swiftperm
    if _swiftperm is not None:
        return _swiftperm
    from swiftperm import SwiftPerm
    _swiftperm = SwiftPerm(dylib_path)
    return _swiftperm


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def fmt_throughput(n_perm, elapsed):
    t = n_perm / elapsed
    if t >= 1_000_000:
        return f"{t/1_000_000:.2f}M/s"
    elif t >= 1_000:
        return f"{t/1_000:.0f}k/s"
    return f"{t:.0f}/s"


def print_table(dataset_label, results, n_perm):
    print(f"\n{'='*72}")
    print(f"  {dataset_label}  |  {n_perm:,} permutations")
    print(f"{'='*72}")
    print(f"  {'Implementation':<30} {'observed':>10} {'p-value':>9} {'Time':>8}  {'perm/s':>10}  {'speedup':>8}")
    print(f"  {'-'*70}")
    baseline = None
    for label, (observed, p_value, elapsed, note) in results.items():
        if baseline is None:
            baseline = elapsed
            speedup = "—"
        else:
            speedup = f"{baseline/elapsed:.1f}x"
        tp = fmt_throughput(n_perm, elapsed)
        note_str = f"  [{note}]" if note else ""
        print(f"  {label:<30} {observed:>10.6f} {p_value:>9.5f} {elapsed:>7.3f}s  {tp:>10}  {speedup:>8}{note_str}")
    print(f"{'='*72}")


# ---------------------------------------------------------------------------
# Per-dataset benchmark
# ---------------------------------------------------------------------------

def benchmark_dataset(tag, z_path, w_path, n_perm, seed, dylib_path=None, baselines=None):
    z = read_z_vector(z_path)
    n, nnz, rows, cols, values = read_sparse_weights(w_path)
    W = build_sparse_matrix(n, rows, cols, values)

    print(f"  n={n:,}, nnz={nnz:,}")

    results = {}

    # esda baselines (pre-computed, loaded from data/baselines.json)
    if baselines:
        for stat_label in ("esda Moran's I", "esda Geary's C"):
            if stat_label in baselines:
                b = baselines[stat_label]
                results[stat_label] = (b["observed"], b["p_sim"], b["elapsed_s"], f"baseline n_perm={b['n_perm']:,}")

    # NumPy — skip for n > 10,000 (would take several minutes)
    if n <= 10000:
        obs, _, pval, elapsed = moran_perm_numpy(z, W, n_perm=n_perm, seed=seed)
        results["NumPy"] = (obs, pval, elapsed, None)

    # Numba — skip for n > 10,000
    if n <= 10000:
        nb = moran_perm_numba(z, rows, cols, values, n, n_perm=n_perm, seed=seed)
        if nb:
            obs_nb, _, pval_nb, elapsed_nb = nb
            results["Numba (parallel)"] = (obs_nb, pval_nb, elapsed_nb, None)

    # Swift: serial, parallel, Metal — via direct ctypes binding, for each statistic
    if dylib_path is not False:
        try:
            sp = _get_swiftperm(dylib_path)

            for stat, label in [("moran", "Moran's I"), ("gearysc", "Geary's C")]:
                r = sp.perm_serial(z, rows, cols, values, n, n_perm=n_perm, seed=seed,
                                   statistic=stat)
                results[f"Swift serial ({label})"] = (r.observed, r.p_value, r.elapsed_seconds, None)

                r = sp.perm_parallel(z, rows, cols, values, n, n_perm=n_perm, seed=seed,
                                     statistic=stat)
                results[f"Swift parallel ({label})"] = (r.observed, r.p_value, r.elapsed_seconds, None)

                try:
                    r = sp.perm_metal(z, rows, cols, values, n, n_perm=n_perm, seed=seed,
                                      statistic=stat, fallback_to_parallel=False)
                    results[f"Swift+Metal ({label})"] = (r.observed, r.p_value, r.elapsed_seconds, None)
                except RuntimeError:
                    pass  # Metal unavailable

        except OSError:
            pass  # dylib not built — skip Swift rows silently

    return results, n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dylib", type=str, nargs="?", const=True, default=None,
                        help="Path to libSwiftGeoLib.dylib (omit path to use default location)")
    parser.add_argument("--n-perm", type=int, default=N_PERM)
    args = parser.parse_args()

    n_perm = args.n_perm
    dylib_path = args.dylib if args.dylib is not True else None
    RESULTS_DIR.mkdir(exist_ok=True)

    # Load pre-computed esda baselines if available
    baselines_path = DATA_DIR / "baselines.json"
    all_baselines = {}
    if baselines_path.exists():
        with open(baselines_path) as f:
            bl = json.load(f)
        all_baselines = bl.get("datasets", {})
        print(f"Loaded esda baselines from {baselines_path} "
              f"(generated {bl.get('generated_at', '?')[:10]})")
    else:
        print(f"No baselines found — run baseline.py to generate esda comparison rows")

    datasets = [
        ("Columbus_n49",         "Columbus (n=49)",                 "columbus_z.bin",      "columbus_weights.bin"),
        ("KingCounty_n21613",    "King County WA (n=21,613)",       "kingcounty_z.bin",    "kingcounty_weights.bin"),
        ("NCOVR_n3085",          "NCOVR US Counties (n=3,085)",     "ncovr_z.bin",         "ncovr_weights.bin"),
        ("SDOH_n71901",          "US SDOH Tracts (n=71,901)",       "sdoh_z.bin",          "sdoh_weights.bin"),
        ("NYC_Earnings_n108487", "NYC Earnings Blocks (n=108,487)", "nyc_earnings_z.bin",  "nyc_earnings_weights.bin"),
    ]

    print(f"\nRunning benchmarks: {n_perm:,} permutations each\n")

    all_results = {}

    for tag, label, z_file, w_file in datasets:
        z_path = DATA_DIR / z_file
        w_path = DATA_DIR / w_file
        if not z_path.exists():
            print(f"\n[{label}] fixtures not found — run generate_fixtures.py")
            continue

        print(f"\n{'─'*72}")
        print(f"  Dataset: {label}")
        results, n = benchmark_dataset(
            tag, z_path, w_path, n_perm, SEED,
            dylib_path=dylib_path,
            baselines=all_baselines.get(label),
        )
        print_table(label, results, n_perm)
        all_results[label] = {
            k: {"observed": v[0], "p_value": v[1],
                "elapsed_s": v[2], "throughput_perm_s": n_perm / v[2],
                "note": v[3]}
            for k, v in results.items()
        }

    summary = {"n_permutations": n_perm, "datasets": all_results}
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR}/summary.json")


if __name__ == "__main__":
    main()
