#!/usr/bin/env python3
"""
generate_fixtures.py
Generate binary fixtures from the Columbus dataset for the Swift benchmark.

Writes:
  ../data/columbus_z.bin      -- standardized HOVAL variable
  ../data/columbus_weights.bin -- row-standardized Queen weights (COO)
  ../data/columbus_meta.json  -- human-readable metadata

Usage:
  uv run python generate_fixtures.py
  # or: python generate_fixtures.py  (with libpysal installed)
"""

import struct
import json
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def write_z_vector(z: np.ndarray, path: Path):
    """Write z vector: [int32 n][float64 x n]"""
    with open(path, "wb") as f:
        f.write(struct.pack("<i", len(z)))
        f.write(z.astype(np.float64).tobytes())
    print(f"  Wrote {path.name}: n={len(z)}")


def write_sparse_weights(rows, cols, values, n: int, path: Path):
    """Write COO weights: [int32 n][int32 nnz][int32 x nnz rows][int32 x nnz cols][float64 x nnz values]"""
    nnz = len(values)
    with open(path, "wb") as f:
        f.write(struct.pack("<i", n))
        f.write(struct.pack("<i", nnz))
        f.write(np.array(rows, dtype=np.int32).tobytes())
        f.write(np.array(cols, dtype=np.int32).tobytes())
        f.write(np.array(values, dtype=np.float64).tobytes())
    print(f"  Wrote {path.name}: n={n}, nnz={nnz}")


def main():
    try:
        import libpysal
        from libpysal.weights import Queen
        from libpysal import examples
    except ImportError:
        print("ERROR: libpysal not found. Install with: pip install libpysal")
        raise

    print("Loading Columbus dataset...")
    shp_path = examples.get_path("columbus.shp")

    # Build Queen contiguity weights
    w = Queen.from_shapefile(shp_path)
    w.transform = "r"  # row-standardize
    n = w.n
    print(f"  n={n} observations")

    # Load HOVAL (home values) as our test variable
    import geopandas as gpd
    gdf = gpd.read_file(shp_path)
    y = gdf["HOVAL"].values.astype(np.float64)

    # Standardize
    z = (y - y.mean()) / y.std()
    print(f"  z: mean={z.mean():.6f}, std={z.std():.6f}")

    # Convert weights to COO arrays (integer indices)
    id_order = w.id_order
    id_to_idx = {id_: i for i, id_ in enumerate(id_order)}

    rows, cols, values = [], [], []
    for focal_id, neighbors in w.neighbors.items():
        i = id_to_idx[focal_id]
        for neighbor_id, wval in zip(neighbors, w.weights[focal_id]):
            j = id_to_idx[neighbor_id]
            rows.append(i)
            cols.append(j)
            values.append(wval)

    # Compute reference Moran's I using esda for verification
    try:
        from esda.moran import Moran
        mi = Moran(y, w, permutations=0)
        ref_I = mi.I
        print(f"  Reference Moran's I (esda): {ref_I:.6f}")
    except ImportError:
        # Manual calculation for verification
        wz = np.zeros(n)
        for k in range(len(values)):
            wz[rows[k]] += values[k] * z[cols[k]]
        ref_I = np.dot(z, wz) / n
        print(f"  Reference Moran's I (manual): {ref_I:.6f}")

    # Write fixtures
    DATA_DIR.mkdir(exist_ok=True)
    print("\nWriting fixtures...")
    write_z_vector(z, DATA_DIR / "columbus_z.bin")
    write_sparse_weights(rows, cols, values, n, DATA_DIR / "columbus_weights.bin")

    # Metadata for reference
    meta = {
        "dataset": "Columbus OH (libpysal example)",
        "variable": "HOVAL (home values, standardized)",
        "n": n,
        "nnz": len(values),
        "weights_type": "Queen contiguity, row-standardized",
        "reference_morans_i": float(ref_I),
        "z_mean": float(z.mean()),
        "z_std": float(z.std()),
    }
    with open(DATA_DIR / "columbus_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Wrote columbus_meta.json")
    print(f"\nReference Moran's I = {ref_I:.6f}")
    print("Fixtures ready.")


if __name__ == "__main__":
    main()
