#!/usr/bin/env python3
"""
generate_synthetic.py
Generate larger synthetic datasets for stress-testing the permutation benchmark.

Creates random point datasets with KNN spatial weights at multiple scales:
  - n=500,   k=8
  - n=2000,  k=8
  - n=5000,  k=8
  - n=10000, k=8

Writes binary fixtures to ../data/synthetic_<n>_z.bin and ../data/synthetic_<n>_weights.bin

Usage:
  uv run python generate_synthetic.py
"""

import struct
import json
import numpy as np
from pathlib import Path
from scipy.spatial import KDTree

DATA_DIR = Path(__file__).parent.parent / "data"
SIZES = [500, 2000, 5000, 10000]
K = 8
SEED = 42


def write_z_vector(z: np.ndarray, path: Path):
    with open(path, "wb") as f:
        f.write(struct.pack("<i", len(z)))
        f.write(z.astype(np.float64).tobytes())


def write_sparse_weights(rows, cols, values, n: int, path: Path):
    nnz = len(values)
    with open(path, "wb") as f:
        f.write(struct.pack("<i", n))
        f.write(struct.pack("<i", nnz))
        f.write(np.array(rows, dtype=np.int32).tobytes())
        f.write(np.array(cols, dtype=np.int32).tobytes())
        f.write(np.array(values, dtype=np.float64).tobytes())


def build_knn_weights(coords: np.ndarray, k: int):
    """Build row-standardized KNN weights using KDTree."""
    n = len(coords)
    tree = KDTree(coords)

    # k+1 because query returns self as nearest neighbor
    distances, indices = tree.query(coords, k=k + 1)

    rows, cols, values = [], [], []
    for i in range(n):
        neighbors = indices[i, 1:]  # exclude self
        w = 1.0 / k                 # uniform, row-standardized
        for j in neighbors:
            rows.append(i)
            cols.append(j)
            values.append(w)

    return (np.array(rows, dtype=np.int32),
            np.array(cols, dtype=np.int32),
            np.array(values, dtype=np.float64))


def moran_i_reference(z, rows, cols, values, n):
    """Compute reference Moran's I for verification."""
    wz = np.zeros(n)
    for k in range(len(values)):
        wz[rows[k]] += values[k] * z[cols[k]]
    return float(z @ wz) / n


def main():
    DATA_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)

    print(f"Generating synthetic datasets (k={K} KNN weights)\n")
    meta_all = {}

    for n in SIZES:
        print(f"  n={n:,}...")

        # Random points in unit square
        coords = rng.random((n, 2))

        # Random spatially autocorrelated variable via spatial smoothing
        # Start with noise, smooth with KNN average to introduce autocorrelation
        y_raw = rng.standard_normal(n)
        tree = KDTree(coords)
        _, indices = tree.query(coords, k=K + 1)
        y_smooth = np.array([y_raw[indices[i, 1:]].mean() for i in range(n)])
        y = y_raw * 0.3 + y_smooth * 0.7  # blend for moderate autocorrelation

        # Standardize
        z = (y - y.mean()) / y.std()

        # Build KNN weights
        rows, cols, values = build_knn_weights(coords, K)
        nnz = len(values)

        # Reference Moran's I
        ref_I = moran_i_reference(z, rows, cols, values, n)
        print(f"    nnz={nnz:,}, Moran's I={ref_I:.6f}")

        # Write fixtures
        z_path = DATA_DIR / f"synthetic_{n}_z.bin"
        w_path = DATA_DIR / f"synthetic_{n}_weights.bin"
        write_z_vector(z, z_path)
        write_sparse_weights(rows, cols, values, n, w_path)

        meta_all[str(n)] = {
            "n": n,
            "k": K,
            "nnz": nnz,
            "weights_type": f"KNN k={K}, row-standardized",
            "reference_morans_i": float(ref_I),
        }

    # Write combined metadata
    with open(DATA_DIR / "synthetic_meta.json", "w") as f:
        json.dump(meta_all, f, indent=2)

    print(f"\nWrote {len(SIZES)} datasets to {DATA_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
