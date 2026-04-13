#!/usr/bin/env python3
"""
benchmark.py
Run Moran's I permutation test across multiple dataset sizes.
Compares NumPy, Numba (parallel), and Swift (Accelerate, serial+parallel).

Usage:
  uv run python benchmark.py                          # Columbus only, Python
  uv run python benchmark.py --synthetic              # Add synthetic datasets
  uv run python benchmark.py --synthetic --swift-bin ../SwiftGeo/.build/release/SwiftGeoCLI
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
        n_perm   = struct.unpack("<i", f.read(4))[0]
        null     = np.frombuffer(f.read(8 * n_perm), dtype=np.float64).copy()
        observed = struct.unpack("<d", f.read(8))[0]
        p_value  = struct.unpack("<d", f.read(8))[0]
        elapsed  = struct.unpack("<d", f.read(8))[0]
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
# Swift runner
# ---------------------------------------------------------------------------

def run_swift(swift_bin, z_path, w_path, out_path, n_perm, seed):
    cmd = [swift_bin, str(z_path), str(w_path), str(out_path), str(n_perm), str(seed)]
    subprocess.run(cmd, check=True, capture_output=False)
    return read_swift_results(out_path)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def fmt_row(label, observed, p_value, elapsed, n_perm, baseline_elapsed=None):
    throughput = n_perm / elapsed
    speedup = f"{baseline_elapsed/elapsed:.1f}x" if baseline_elapsed else "—"
    return (f"{label:<26} {observed:>10.6f} {p_value:>9.5f} "
            f"{elapsed:>9.3f}  {throughput:>12,.0f}  {speedup:>6}")


def print_table(rows, n_perm, dataset_label):
    print(f"\n{'='*80}")
    print(f"  {dataset_label}  |  {n_perm:,} permutations")
    print(f"{'='*80}")
    print(f"{'Implementation':<26} {'Moran I':>10} {'p-value':>9} {'Time(s)':>9}  {'perm/s':>12}  {'vs NumPy':>6}")
    print(f"{'-'*80}")
    for row in rows:
        print(row)
    print(f"{'='*80}")


# ---------------------------------------------------------------------------
# Per-dataset benchmark
# ---------------------------------------------------------------------------

def benchmark_dataset(label, z_path, w_path, n_perm, seed, swift_bin=None):
    z = read_z_vector(z_path)
    n, nnz, rows, cols, values = read_sparse_weights(w_path)
    W = build_sparse_matrix(n, rows, cols, values)

    print(f"\n  n={n:,}, nnz={nnz:,}")
    results = {}

    # NumPy
    obs_np, _, pval_np, elapsed_np = moran_perm_numpy(z, W, n_perm=n_perm, seed=seed)
    results["NumPy"] = (obs_np, pval_np, elapsed_np)
    print(f"    NumPy:  {elapsed_np:.3f}s  ({n_perm/elapsed_np:,.0f} perm/s)")

    # Numba
    nb = moran_perm_numba(z, rows, cols, values, n, n_perm=n_perm, seed=seed)
    if nb:
        obs_nb, _, pval_nb, elapsed_nb = nb
        results["Numba (parallel)"] = (obs_nb, pval_nb, elapsed_nb)
        print(f"    Numba:  {elapsed_nb:.3f}s  ({n_perm/elapsed_nb:,.0f} perm/s)")

    # Swift
    if swift_bin:
        swift_out = RESULTS_DIR / f"swift_{label}_null.bin"
        _, obs_sw, pval_sw, elapsed_sw, n_threads = run_swift(
            swift_bin, z_path, w_path, swift_out, n_perm, seed
        )
        results[f"Swift parallel ({n_threads}t)"] = (obs_sw, pval_sw, elapsed_sw)
        print(f"    Swift:  {elapsed_sw:.3f}s  ({n_perm/elapsed_sw:,.0f} perm/s)")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--swift-bin", type=str, default=None)
    parser.add_argument("--synthetic", action="store_true",
                        help="Include synthetic datasets at multiple scales")
    parser.add_argument("--n-perm", type=int, default=N_PERM)
    args = parser.parse_args()

    n_perm = args.n_perm
    RESULTS_DIR.mkdir(exist_ok=True)

    # Suppress Swift stdout from cluttering our output
    # (Swift prints its own progress; we capture it separately if needed)

    datasets = [("Columbus (n=49)", "columbus_z.bin", "columbus_weights.bin")]

    if args.synthetic:
        with open(DATA_DIR / "synthetic_meta.json") as f:
            meta = json.load(f)
        for n_str in sorted(meta.keys(), key=int):
            n = int(n_str)
            datasets.append((
                f"Synthetic KNN (n={n:,})",
                f"synthetic_{n}_z.bin",
                f"synthetic_{n}_weights.bin"
            ))

    all_results = {}
    print(f"\nRunning benchmarks: {n_perm:,} permutations each")

    for label, z_file, w_file in datasets:
        print(f"\n{'─'*60}")
        print(f"  Dataset: {label}")
        z_path = DATA_DIR / z_file
        w_path = DATA_DIR / w_file
        if not z_path.exists():
            print(f"  [skipping — fixtures not found, run generate_synthetic.py]")
            continue

        results = benchmark_dataset(
            label.replace(" ", "_").replace(",", "").replace("(", "").replace(")", ""),
            z_path, w_path, n_perm, SEED,
            swift_bin=args.swift_bin
        )
        all_results[label] = results

        # Print table for this dataset
        baseline = results["NumPy"][2]
        rows = [fmt_row(k, v[0], v[1], v[2], n_perm,
                        baseline if k != "NumPy" else None)
                for k, v in results.items()]
        print_table(rows, n_perm, label)

    # Save full summary
    summary = {
        "n_permutations": n_perm,
        "datasets": {
            label: {
                impl: {
                    "observed": v[0],
                    "p_value": v[1],
                    "elapsed_s": v[2],
                    "throughput_perm_s": n_perm / v[2]
                }
                for impl, v in results.items()
            }
            for label, results in all_results.items()
        }
    }
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR}/summary.json")


if __name__ == "__main__":
    main()
