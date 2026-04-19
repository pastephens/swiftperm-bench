#!/usr/bin/env python3
"""
baseline.py
Run native esda/libpysal implementations once and save timing results to data/baselines.json.
These baselines are loaded by benchmark.py to show esda comparison rows and compute speedups.

Statistics covered:
  - Global Moran's I  (esda.Moran)
  - Global Geary's C  (esda.Geary)
  - Local Moran's I   (esda.Moran_Local, n_jobs=-1)
  - Local Geary's C   (esda.Geary_Local, n_jobs=-1)

Usage:
  uv run python baseline.py                        # 9,999 perms, skip local for n > 25,000
  uv run python baseline.py --n-perm 999           # quick test run
  uv run python baseline.py --skip-local-above 0   # global stats only
"""

import argparse
import json
import struct
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

DATA_DIR = Path(__file__).parent.parent / "data"

DEFAULT_N_PERM           = 9999
DEFAULT_SKIP_LOCAL_ABOVE = 25_000


# ---------------------------------------------------------------------------
# Binary I/O  (same format as benchmark.py)
# ---------------------------------------------------------------------------

def read_z_vector(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        n = struct.unpack("<i", f.read(4))[0]
        return np.frombuffer(f.read(8 * n), dtype=np.float64).copy()


def read_sparse_weights(path: Path):
    with open(path, "rb") as f:
        n   = struct.unpack("<i", f.read(4))[0]
        nnz = struct.unpack("<i", f.read(4))[0]
        rows = np.frombuffer(f.read(4 * nnz), dtype=np.int32).copy()
        cols = np.frombuffer(f.read(4 * nnz), dtype=np.int32).copy()
        vals = np.frombuffer(f.read(8 * nnz), dtype=np.float64).copy()
    return n, nnz, rows, cols, vals


# ---------------------------------------------------------------------------
# Weight reconstruction
# ---------------------------------------------------------------------------

def coo_to_w(n, rows, cols, vals):
    """Reconstruct a libpysal W from COO arrays via WSP (zero data round-trip)."""
    from libpysal.weights import WSP
    sp = csr_matrix((vals, (rows, cols)), shape=(n, n))
    return WSP(sp).to_W()


# ---------------------------------------------------------------------------
# Per-dataset baseline
# ---------------------------------------------------------------------------

DATASETS = [
    ("Columbus (n=49)",           "columbus_z.bin",      "columbus_weights.bin"),
    ("NCOVR (n=3,085)",           "ncovr_z.bin",         "ncovr_weights.bin"),
    ("King County WA (n=21,613)", "kingcounty_z.bin",    "kingcounty_weights.bin"),
    ("US SDOH (n=71,901)",        "sdoh_z.bin",          "sdoh_weights.bin"),
    ("NYC Earnings (n=108,487)",  "nyc_earnings_z.bin",  "nyc_earnings_weights.bin"),
]


def baseline_dataset(label, z_path, w_path, n_perm, skip_local_above):
    import esda

    z = read_z_vector(z_path)
    n, nnz, rows, cols, vals = read_sparse_weights(w_path)

    print(f"  Building W (n={n:,}, nnz={nnz:,})...", flush=True)
    w = coo_to_w(n, rows, cols, vals)

    results = {}

    # Global Moran's I
    print(f"  esda.Moran ...", flush=True)
    t0 = time.perf_counter()
    mi = esda.Moran(z, w, permutations=n_perm, two_tailed=False)
    results["esda Moran's I"] = {
        "observed": float(mi.I), "p_sim": float(mi.p_sim),
        "elapsed_s": time.perf_counter() - t0, "n_perm": n_perm,
    }
    _print_row("esda Moran's I", results["esda Moran's I"])

    # Global Geary's C
    print(f"  esda.Geary ...", flush=True)
    t0 = time.perf_counter()
    gc = esda.Geary(z, w, permutations=n_perm)
    results["esda Geary's C"] = {
        "observed": float(gc.C), "p_sim": float(gc.p_sim),
        "elapsed_s": time.perf_counter() - t0, "n_perm": n_perm,
    }
    _print_row("esda Geary's C", results["esda Geary's C"])

    if n <= skip_local_above:
        # Local Moran's I  (n_jobs=-1 → all cores, same as our parallel backend)
        print(f"  esda.Moran_Local (n_jobs=-1) ...", flush=True)
        t0 = time.perf_counter()
        lm = esda.Moran_Local(z, w, permutations=n_perm, n_jobs=-1)
        results["esda Local Moran's I"] = {
            "elapsed_s": time.perf_counter() - t0, "n_perm": n_perm,
            "mean_observed": float(np.mean(lm.Is)),
            "mean_p_sim": float(np.mean(lm.p_sim)),
        }
        _print_row("esda Local Moran's I", results["esda Local Moran's I"])

        # Local Geary's C  (n_jobs=-1)
        print(f"  esda.Geary_Local (n_jobs=-1) ...", flush=True)
        t0 = time.perf_counter()
        lg = esda.Geary_Local(connectivity=w, permutations=n_perm, n_jobs=-1).fit(z)
        results["esda Local Geary's C"] = {
            "elapsed_s": time.perf_counter() - t0, "n_perm": n_perm,
            "mean_observed": float(np.mean(lg.localG)),
            "mean_p_sim": float(np.mean(lg.p_sim)),
        }
        _print_row("esda Local Geary's C", results["esda Local Geary's C"])
    else:
        print(f"  Skipping local stats (n={n:,} > {skip_local_above:,})")

    return results, n


def _print_row(label, r):
    elapsed = r["elapsed_s"]
    n_perm  = r["n_perm"]
    tp = n_perm / elapsed
    tp_str = f"{tp/1e6:.2f}M/s" if tp >= 1e6 else f"{tp/1e3:.0f}k/s"
    obs = r.get("observed", r.get("mean_observed", float("nan")))
    print(f"    {label:<32}  obs={obs:.6f}  {elapsed:.3f}s  {tp_str}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-perm", type=int, default=DEFAULT_N_PERM)
    parser.add_argument("--skip-local-above", type=int, default=DEFAULT_SKIP_LOCAL_ABOVE,
                        help="Skip local perm tests for datasets with n above this threshold")
    args = parser.parse_args()

    n_perm           = args.n_perm
    skip_local_above = args.skip_local_above

    print(f"\nGenerating esda baselines — {n_perm:,} permutations each")
    print(f"Local stats skipped for n > {skip_local_above:,}\n")

    all_results = {}

    for label, z_file, w_file in DATASETS:
        z_path = DATA_DIR / z_file
        w_path = DATA_DIR / w_file
        if not z_path.exists():
            print(f"[{label}] fixtures not found — run generate_fixtures.py\n")
            continue

        print(f"{'─'*60}")
        print(f"  {label}")
        results, n = baseline_dataset(label, z_path, w_path, n_perm, skip_local_above)
        all_results[label] = results
        print()

    out = {
        "n_permutations": n_perm,
        "skip_local_above": skip_local_above,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "datasets": all_results,
    }
    out_path = DATA_DIR / "baselines.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Baselines saved to {out_path}")


if __name__ == "__main__":
    main()
