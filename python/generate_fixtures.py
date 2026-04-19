#!/usr/bin/env python3
"""
generate_fixtures.py
Generate binary fixtures from real datasets for the Swift benchmark.

Writes:
  ../data/columbus_z.bin         -- standardized HOVAL variable
  ../data/columbus_weights.bin   -- row-standardized Queen weights (COO)
  ../data/columbus_meta.json     -- human-readable metadata
  ../data/kingcounty_z.bin       -- standardized log(price) variable
  ../data/kingcounty_weights.bin -- row-standardized KNN-8 weights (COO)
  ../data/kingcounty_meta.json   -- human-readable metadata

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

    # -----------------------------------------------------------------------
    # King County home sales (n=21,613 point observations)
    # -----------------------------------------------------------------------
    print("\nLoading King County dataset...")
    from libpysal.examples import load_example
    from libpysal.weights import KNN

    kc_ex = load_example("Home Sales")
    kc_shp = [f for f in kc_ex.get_file_list() if f.endswith("kc_house.shp") and "__MACOSX" not in f][0]
    kc_gdf = gpd.read_file(kc_shp)
    kc_n = len(kc_gdf)
    print(f"  n={kc_n} observations")

    # Build KNN-8 weights from point coordinates (consistent with synthetic datasets)
    kc_w = KNN.from_dataframe(kc_gdf, k=8)
    kc_w.transform = "r"

    # Log-price as the test variable (prices are log-normally distributed)
    kc_price = kc_gdf["price"].values.astype(np.float64)
    kc_y = np.log(kc_price)
    kc_z = (kc_y - kc_y.mean()) / kc_y.std()
    print(f"  z: mean={kc_z.mean():.6f}, std={kc_z.std():.6f}")

    # Convert to COO
    kc_id_order = kc_w.id_order
    kc_id_to_idx = {id_: i for i, id_ in enumerate(kc_id_order)}
    kc_rows, kc_cols, kc_values = [], [], []
    for focal_id, neighbors in kc_w.neighbors.items():
        i = kc_id_to_idx[focal_id]
        for neighbor_id, wval in zip(neighbors, kc_w.weights[focal_id]):
            j = kc_id_to_idx[neighbor_id]
            kc_rows.append(i)
            kc_cols.append(j)
            kc_values.append(wval)

    try:
        from esda.moran import Moran
        kc_mi = Moran(kc_y, kc_w, permutations=0)
        kc_ref_I = kc_mi.I
        print(f"  Reference Moran's I (esda): {kc_ref_I:.6f}")
    except ImportError:
        kc_wz = np.zeros(kc_n)
        for k in range(len(kc_values)):
            kc_wz[kc_rows[k]] += kc_values[k] * kc_z[kc_cols[k]]
        kc_ref_I = np.dot(kc_z, kc_wz) / kc_n
        print(f"  Reference Moran's I (manual): {kc_ref_I:.6f}")

    print("\nWriting King County fixtures...")
    write_z_vector(kc_z, DATA_DIR / "kingcounty_z.bin")
    write_sparse_weights(kc_rows, kc_cols, kc_values, kc_n, DATA_DIR / "kingcounty_weights.bin")

    kc_meta = {
        "dataset": "King County WA Home Sales 2014-15 (libpysal example)",
        "variable": "log(price), standardized",
        "n": kc_n,
        "nnz": len(kc_values),
        "weights_type": "KNN-8, row-standardized",
        "reference_morans_i": float(kc_ref_I),
        "z_mean": float(kc_z.mean()),
        "z_std": float(kc_z.std()),
    }
    with open(DATA_DIR / "kingcounty_meta.json", "w") as f:
        json.dump(kc_meta, f, indent=2)
    print(f"  Wrote kingcounty_meta.json")

    # -----------------------------------------------------------------------
    # Helper: convert any libpysal W to COO arrays
    # -----------------------------------------------------------------------
    def w_to_coo(w):
        id_order = w.id_order
        id_to_idx = {id_: i for i, id_ in enumerate(id_order)}
        rs, cs, vs = [], [], []
        for focal_id, neighbors in w.neighbors.items():
            i = id_to_idx[focal_id]
            for neighbor_id, wval in zip(neighbors, w.weights[focal_id]):
                rs.append(i)
                cs.append(id_to_idx[neighbor_id])
                vs.append(wval)
        return rs, cs, vs

    def standardize(y):
        z = (y - y.mean()) / y.std()
        return z

    def morans_i_manual(z, rs, cs, vs, n):
        wz = np.zeros(n)
        for k in range(len(vs)):
            wz[rs[k]] += vs[k] * z[cs[k]]
        return float(np.dot(z, wz) / n)

    def write_dataset(tag, label, gdf, y, w, weights_type, var_label):
        n = len(gdf)
        z = standardize(y)
        rs, cs, vs = w_to_coo(w)
        try:
            from esda.moran import Moran
            ref_I = Moran(y, w, permutations=0).I
            print(f"  Reference Moran's I (esda): {ref_I:.6f}")
        except ImportError:
            ref_I = morans_i_manual(z, rs, cs, vs, n)
            print(f"  Reference Moran's I (manual): {ref_I:.6f}")
        print(f"\nWriting {label} fixtures...")
        write_z_vector(z, DATA_DIR / f"{tag}_z.bin")
        write_sparse_weights(rs, cs, vs, n, DATA_DIR / f"{tag}_weights.bin")
        meta = {
            "dataset": label,
            "variable": var_label,
            "n": n,
            "nnz": len(vs),
            "weights_type": weights_type,
            "reference_morans_i": float(ref_I),
            "z_mean": float(z.mean()),
            "z_std": float(z.std()),
        }
        with open(DATA_DIR / f"{tag}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  Wrote {tag}_meta.json")

    # -----------------------------------------------------------------------
    # NCOVR — US counties, homicide rate 1990, Queen contiguity
    # -----------------------------------------------------------------------
    print("\nLoading NCOVR dataset...")
    from libpysal.weights import Queen
    ncovr_ex = load_example("NCOVR")
    ncovr_shp = [f for f in ncovr_ex.get_file_list() if f.endswith(".shp") and "__MACOSX" not in f][0]
    ncovr_gdf = gpd.read_file(ncovr_shp)
    print(f"  n={len(ncovr_gdf):,} observations")
    ncovr_w = Queen.from_dataframe(ncovr_gdf)
    ncovr_w.transform = "r"
    write_dataset("ncovr", "NCOVR US Counties (n=3,085)",
                  ncovr_gdf, ncovr_gdf["HR90"].values.astype(np.float64),
                  ncovr_w, "Queen contiguity, row-standardized", "HR90 (homicide rate 1990), standardized")

    # -----------------------------------------------------------------------
    # US SDOH — census tracts, poverty rate, KNN-8 from centroids
    # -----------------------------------------------------------------------
    print("\nLoading US SDOH dataset...")
    sdoh_ex = load_example("US SDOH")
    sdoh_shp = [f for f in sdoh_ex.get_file_list() if f.endswith(".shp") and "__MACOSX" not in f][0]
    sdoh_gdf = gpd.read_file(sdoh_shp)
    print(f"  n={len(sdoh_gdf):,} observations")
    print("  Building KNN-8 weights from centroids (this may take ~30s)...")
    sdoh_w = KNN.from_dataframe(sdoh_gdf, k=8)
    sdoh_w.transform = "r"
    write_dataset("sdoh", "US SDOH Census Tracts (n=71,901)",
                  sdoh_gdf, sdoh_gdf["ep_pov"].values.astype(np.float64),
                  sdoh_w, "KNN-8 from centroids, row-standardized", "ep_pov (poverty rate %), standardized")

    # -----------------------------------------------------------------------
    # NYC Earnings — census blocks, total jobs 2014, KNN-8 from centroids
    # -----------------------------------------------------------------------
    print("\nLoading NYC Earnings dataset...")
    nyc_ex = load_example("NYC Earnings")
    nyc_shp = [f for f in nyc_ex.get_file_list() if f.endswith(".shp") and "__MACOSX" not in f][0]
    nyc_gdf = gpd.read_file(nyc_shp)
    print(f"  n={len(nyc_gdf):,} observations")
    print("  Building KNN-8 weights from centroids (this may take ~60s)...")
    nyc_w = KNN.from_dataframe(nyc_gdf, k=8)
    nyc_w.transform = "r"
    # log1p of total jobs (count variable, right-skewed, many zeros)
    nyc_y = np.log1p(nyc_gdf["C000_14"].values.astype(np.float64))
    write_dataset("nyc_earnings", "NYC Earnings Census Blocks (n=108,487)",
                  nyc_gdf, nyc_y,
                  nyc_w, "KNN-8 from centroids, row-standardized", "log1p(C000_14 total jobs 2014), standardized")

    print("\nAll fixtures ready.")


if __name__ == "__main__":
    main()
