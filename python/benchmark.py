#!/usr/bin/env python3
"""
benchmark.py
Run Moran's I permutation test in Python (Numba + pure NumPy baselines)
and compare against Swift results written by SwiftGeoCLI.

Usage:
  uv run python benchmark.py                        # Python benchmarks only
  uv run python benchmark.py --compare              # Python + read Swift results
  uv run python benchmark.py --swift-bin ../SwiftGeo/.build/release/SwiftGeoCLI
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
# Binary readers (mirror of Swift BinaryIO.swift)
# ---------------------------------------------------------------------------

def read_z_vector(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        n = struct.unpack("<i", f.read(4))[0]
        z = np.frombuffer(f.read(8 * n), dtype=np.float64).copy()
    return z


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
        n_perm = struct.unpack("<i", f.read(4))[0]
        null_dist = np.frombuffer(f.read(8 * n_perm), dtype=np.float64).copy()
        observed  = struct.unpack("<d", f.read(8))[0]
        p_value   = struct.unpack("<d", f.read(8))[0]
        elapsed   = struct.unpack("<d", f.read(8))[0]
    return null_dist, observed, p_value, elapsed


# ---------------------------------------------------------------------------
# Python implementations
# ---------------------------------------------------------------------------

def build_sparse_matrix(n, rows, cols, values):
    from scipy.sparse import csr_matrix
    return csr_matrix((values, (rows, cols)), shape=(n, n))


def moran_i_scipy(z, W_sparse):
    wz = W_sparse.dot(z)
    return float(z @ wz) / len(z)


def moran_perm_numpy(z, W_sparse, n_perm=N_PERM, seed=SEED):
    """Pure NumPy permutation test."""
    rng = np.random.default_rng(seed)
    observed = moran_i_scipy(z, W_sparse)
    null = np.empty(n_perm)

    t0 = time.perf_counter()
    z_perm = z.copy()
    for i in range(n_perm):
        rng.shuffle(z_perm)
        null[i] = moran_i_scipy(z_perm, W_sparse)
    elapsed = time.perf_counter() - t0

    p_value = np.mean(np.abs(null) >= abs(observed))
    return observed, null, p_value, elapsed


def moran_perm_numba(z, rows, cols, values, n, n_perm=N_PERM, seed=SEED):
    """Numba-accelerated permutation test."""
    try:
        from numba import njit, prange
    except ImportError:
        print("  [Numba not available, skipping]")
        return None

    @njit(parallel=False, cache=True)
    def _moran_i(z, rows, cols, values, n):
        wz = np.zeros(n)
        for k in range(len(values)):
            wz[rows[k]] += values[k] * z[cols[k]]
        result = 0.0
        for i in range(n):
            result += z[i] * wz[i]
        return result / n

    @njit(parallel=True, cache=True)
    def _perm_loop(z, rows, cols, values, n, n_perm, seed):
        null = np.empty(n_perm)
        for p in prange(n_perm):
            z_p = z.copy()
            # Per-permutation seeded LCG shuffle
            state = seed + np.uint64(p * 6364136223846793005)
            for i in range(n - 1, 0, -1):
                state = state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
                j = int(state >> np.uint64(33)) % (i + 1)
                z_p[i], z_p[j] = z_p[j], z_p[i]
            null[p] = _moran_i(z_p, rows, cols, values, n)
        return null

    # Warm up JIT
    print("  Warming up Numba JIT...")
    _ = _moran_i(z, rows, cols, values, n)
    _ = _perm_loop(z, rows, cols, values, n, 10, np.uint64(seed))

    observed = _moran_i(z, rows, cols, values, n)

    t0 = time.perf_counter()
    null = _perm_loop(z, rows, cols, values, n, n_perm, np.uint64(seed))
    elapsed = time.perf_counter() - t0

    p_value = float(np.mean(np.abs(null) >= abs(observed)))
    return observed, null, p_value, elapsed


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_result(label, observed, p_value, elapsed, n_perm):
    throughput = n_perm / elapsed
    print(f"\n  [{label}]")
    print(f"    Observed Moran's I : {observed:.6f}")
    print(f"    p-value (2-sided)  : {p_value:.6f}")
    print(f"    Elapsed            : {elapsed:.3f}s")
    print(f"    Throughput         : {throughput:,.0f} perm/s")


def print_comparison(results: dict, n_perm: int):
    print("\n" + "="*60)
    print(f"BENCHMARK SUMMARY  ({n_perm:,} permutations, Columbus dataset)")
    print("="*60)
    print(f"{'Implementation':<22} {'Moran I':>10} {'p-value':>10} {'Time(s)':>10} {'perm/s':>12}")
    print("-"*60)
    baseline_elapsed = None
    for label, (observed, p_value, elapsed) in results.items():
        if baseline_elapsed is None:
            baseline_elapsed = elapsed
        throughput = n_perm / elapsed
        speedup = baseline_elapsed / elapsed
        print(f"{label:<22} {observed:>10.6f} {p_value:>10.6f} {elapsed:>10.3f} {throughput:>12,.0f}  ({speedup:.1f}x)")
    print("="*60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true",
                        help="Also read Swift results from results/")
    parser.add_argument("--swift-bin", type=str, default=None,
                        help="Path to SwiftGeoCLI binary to run automatically")
    parser.add_argument("--n-perm", type=int, default=N_PERM)
    args = parser.parse_args()

    n_perm = args.n_perm
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading fixtures...")
    z = read_z_vector(DATA_DIR / "columbus_z.bin")
    n, nnz, rows, cols, values = read_sparse_weights(DATA_DIR / "columbus_weights.bin")
    print(f"  n={n}, nnz={nnz}")

    with open(DATA_DIR / "columbus_meta.json") as f:
        meta = json.load(f)
    print(f"  Reference Moran's I: {meta['reference_morans_i']:.6f}")

    W_sparse = build_sparse_matrix(n, rows, cols, values)
    results = {}

    # --- NumPy baseline ---
    print("\nRunning NumPy baseline...")
    obs, null, pval, elapsed = moran_perm_numpy(z, W_sparse, n_perm=n_perm)
    print_result("NumPy", obs, pval, elapsed, n_perm)
    results["NumPy"] = (obs, pval, elapsed)
    np.save(RESULTS_DIR / "numpy_null.npy", null)

    # --- Numba ---
    print("\nRunning Numba (parallel)...")
    numba_result = moran_perm_numba(z, rows, cols, values, n, n_perm=n_perm)
    if numba_result is not None:
        obs_nb, null_nb, pval_nb, elapsed_nb = numba_result
        print_result("Numba (parallel)", obs_nb, pval_nb, elapsed_nb, n_perm)
        results["Numba (parallel)"] = (obs_nb, pval_nb, elapsed_nb)
        np.save(RESULTS_DIR / "numba_null.npy", null_nb)

    # --- Swift (run binary if provided) ---
    swift_out = RESULTS_DIR / "swift_null.bin"
    if args.swift_bin:
        print(f"\nRunning Swift ({args.swift_bin})...")
        cmd = [args.swift_bin,
               str(DATA_DIR / "columbus_z.bin"),
               str(DATA_DIR / "columbus_weights.bin"),
               str(swift_out),
               str(n_perm),
               str(SEED)]
        subprocess.run(cmd, check=True)
        args.compare = True

    # --- Read Swift results ---
    if args.compare and swift_out.exists():
        print("\nReading Swift results...")
        null_sw, obs_sw, pval_sw, elapsed_sw = read_swift_results(swift_out)
        print_result("Swift (Accelerate)", obs_sw, pval_sw, elapsed_sw, n_perm)
        results["Swift (Accelerate)"] = (obs_sw, pval_sw, elapsed_sw)

    # --- Summary table ---
    if len(results) > 1:
        print_comparison(results, n_perm)

    # Save summary JSON
    summary = {
        "n_permutations": n_perm,
        "dataset": meta["dataset"],
        "results": {
            k: {"observed": v[0], "p_value": v[1], "elapsed_s": v[2],
                "throughput_perm_s": n_perm / v[2]}
            for k, v in results.items()
        }
    }
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR}/summary.json")


if __name__ == "__main__":
    main()
