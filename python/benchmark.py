#!/usr/bin/env python3
"""
benchmark.py
Run Moran's I permutation test across multiple dataset sizes.
Compares NumPy, Numba (parallel), and Swift (serial, parallel, Metal).

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

# Metal scratch buffer limit: nPerm * n * 4 bytes <= 512MB
METAL_SCRATCH_LIMIT_MB = 512


def metal_safe(n: int, n_perm: int) -> bool:
    scratch_mb = (n_perm * n * 4) / (1024 * 1024)
    return scratch_mb <= METAL_SCRATCH_LIMIT_MB


def metal_scratch_mb(n: int, n_perm: int) -> float:
    return (n_perm * n * 4) / (1024 * 1024)


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
# Swift runner
# ---------------------------------------------------------------------------

def run_swift(swift_bin, z_path, w_path, out_path, n_perm, seed):
    cmd = [swift_bin,
           str(z_path), str(w_path), str(out_path),
           str(n_perm), str(seed)]
    result = subprocess.run(cmd, check=True, capture_output=False)
    return read_swift_results(out_path)


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
    print(f"  {'Implementation':<28} {'Moran I':>10} {'p-value':>9} {'Time':>8}  {'perm/s':>10}  {'speedup':>8}")
    print(f"  {'-'*68}")
    baseline = None
    for label, (observed, p_value, elapsed, note) in results.items():
        if baseline is None:
            baseline = elapsed
        speedup = f"{baseline/elapsed:.1f}x" if label != "NumPy" else "—"
        tp = fmt_throughput(n_perm, elapsed)
        note_str = f"  [{note}]" if note else ""
        print(f"  {label:<28} {observed:>10.6f} {p_value:>9.5f} {elapsed:>7.3f}s  {tp:>10}  {speedup:>8}{note_str}")
    print(f"{'='*72}")


# ---------------------------------------------------------------------------
# Per-dataset benchmark
# ---------------------------------------------------------------------------

def benchmark_dataset(tag, z_path, w_path, n_perm, seed, swift_bin=None):
    z = read_z_vector(z_path)
    n, nnz, rows, cols, values = read_sparse_weights(w_path)
    W = build_sparse_matrix(n, rows, cols, values)

    scratch = metal_scratch_mb(n, n_perm)
    metal_ok = metal_safe(n, n_perm)
    print(f"  n={n:,}, nnz={nnz:,}  |  Metal scratch: {scratch:.0f}MB {'✓' if metal_ok else '✗ (skip)'}")

    # results dict: label -> (observed, p_value, elapsed, note)
    results = {}

    # NumPy
    obs, _, pval, elapsed = moran_perm_numpy(z, W, n_perm=n_perm, seed=seed)
    results["NumPy"] = (obs, pval, elapsed, None)

    # Numba
    nb = moran_perm_numba(z, rows, cols, values, n, n_perm=n_perm, seed=seed)
    if nb:
        obs_nb, _, pval_nb, elapsed_nb = nb
        results["Numba (parallel)"] = (obs_nb, pval_nb, elapsed_nb, None)

    # Swift
    if swift_bin:
        swift_out = RESULTS_DIR / f"swift_{tag}_null.bin"
        _, obs_sw, pval_sw, elapsed_sw, n_threads = run_swift(
            swift_bin, z_path, w_path, swift_out, n_perm, seed
        )
        # Swift CLI now runs serial + parallel + Metal internally and writes
        # the best available result. We label it based on Metal availability.
        metal_label = f"Swift+Metal" if metal_ok else f"Swift parallel ({n_threads}t)"
        note = None if metal_ok else "Metal skipped — scratch too large"
        results[metal_label] = (obs_sw, pval_sw, elapsed_sw, note)

    return results, n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--swift-bin", type=str, default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--n-perm", type=int, default=N_PERM)
    args = parser.parse_args()

    n_perm = args.n_perm
    RESULTS_DIR.mkdir(exist_ok=True)

    datasets = [("Columbus_n49", "Columbus (n=49)", "columbus_z.bin", "columbus_weights.bin")]

    if args.synthetic:
        with open(DATA_DIR / "synthetic_meta.json") as f:
            meta = json.load(f)
        for n_str in sorted(meta.keys(), key=int):
            n = int(n_str)
            tag = f"Synthetic_KNN_n{n}"
            datasets.append((
                tag,
                f"Synthetic KNN (n={n:,})",
                f"synthetic_{n}_z.bin",
                f"synthetic_{n}_weights.bin"
            ))

    print(f"\nRunning benchmarks: {n_perm:,} permutations each")
    print(f"Metal scratch limit: {METAL_SCRATCH_LIMIT_MB}MB\n")

    all_results = {}

    for tag, label, z_file, w_file in datasets:
        z_path = DATA_DIR / z_file
        w_path = DATA_DIR / w_file
        if not z_path.exists():
            print(f"\n[{label}] fixtures not found — run generate_synthetic.py")
            continue

        print(f"\n{'─'*72}")
        print(f"  Dataset: {label}")
        results, n = benchmark_dataset(
            tag, z_path, w_path, n_perm, SEED, swift_bin=args.swift_bin
        )
        print_table(label, results, n_perm)
        all_results[label] = {
            k: {"observed": v[0], "p_value": v[1],
                "elapsed_s": v[2], "throughput_perm_s": n_perm / v[2],
                "note": v[3]}
            for k, v in results.items()
        }

    # Save summary
    summary = {"n_permutations": n_perm, "datasets": all_results}
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR}/summary.json")


if __name__ == "__main__":
    main()
