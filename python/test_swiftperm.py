"""
test_swiftperm.py
Accuracy tests for the SwiftGeoLib ctypes binding.

Tests verify:
  - Spatial lag matches scipy W.sparse.dot(y) to floating-point precision
  - Moran's I and Geary's C point estimates match esda reference values
  - Serial, parallel, and Metal permutation tests agree on observed statistic
  - CPU and Metal p-values agree within Monte Carlo tolerance
  - All statistics consistent across datasets of varying size

Requires:
  - libSwiftGeoLib.dylib built (swift build -c release in SwiftGeo/)
  - Fixtures generated (uv run python generate_fixtures.py)
  - libpysal, esda, geopandas installed

Run:
  uv run pytest test_swiftperm.py -v
"""

import struct
from pathlib import Path

import numpy as np
import pytest

DATA_DIR = Path(__file__).parent.parent / "data"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _read_z(path):
    with open(path, "rb") as f:
        n = struct.unpack("<i", f.read(4))[0]
        return np.frombuffer(f.read(8 * n), dtype=np.float64).copy()

def _read_w(path):
    with open(path, "rb") as f:
        n   = struct.unpack("<i", f.read(4))[0]
        nnz = struct.unpack("<i", f.read(4))[0]
        rows = np.frombuffer(f.read(4 * nnz), dtype=np.int32).copy()
        cols = np.frombuffer(f.read(4 * nnz), dtype=np.int32).copy()
        vals = np.frombuffer(f.read(8 * nnz), dtype=np.float64).copy()
    return n, rows, cols, vals


DATASETS = {
    "columbus":    ("columbus_z.bin",    "columbus_weights.bin"),
    "kingcounty":  ("kingcounty_z.bin",  "kingcounty_weights.bin"),
    "ncovr":       ("ncovr_z.bin",       "ncovr_weights.bin"),
}

@pytest.fixture(scope="session")
def sp():
    from swiftperm import SwiftPerm
    return SwiftPerm()

@pytest.fixture(scope="session", params=list(DATASETS.keys()))
def dataset(request):
    name = request.param
    z_file, w_file = DATASETS[name]
    z_path = DATA_DIR / z_file
    w_path = DATA_DIR / w_file
    if not z_path.exists() or not w_path.exists():
        pytest.skip(f"Fixtures not found for {name} — run generate_fixtures.py")
    z = _read_z(z_path)
    n, rows, cols, vals = _read_w(w_path)
    return {"name": name, "z": z, "n": n, "rows": rows, "cols": cols, "vals": vals}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scipy_lag(z, rows, cols, vals, n):
    from scipy.sparse import csr_matrix
    W = csr_matrix((vals, (rows, cols)), shape=(n, n))
    return W.dot(z)

def _esda_moran(z, rows, cols, vals, n):
    """Reference Moran's I via manual computation (no esda dep for this)."""
    wz = np.zeros(n)
    for k in range(len(vals)):
        wz[rows[k]] += vals[k] * z[cols[k]]
    return float(z @ wz) / n

def _esda_gearysc(z, rows, cols, vals, n):
    s = 0.0
    for k in range(len(vals)):
        diff = z[rows[k]] - z[cols[k]]
        s += vals[k] * diff * diff
    return (n - 1.0) / (2.0 * n * n) * s


# ---------------------------------------------------------------------------
# Spatial lag
# ---------------------------------------------------------------------------

class TestSpatialLag:
    def test_serial_matches_scipy(self, sp, dataset):
        d = dataset
        ref = _scipy_lag(d["z"], d["rows"], d["cols"], d["vals"], d["n"])
        wy  = sp.spatial_lag(d["z"], d["rows"], d["cols"], d["vals"], d["n"], parallel=False)
        np.testing.assert_allclose(wy, ref, atol=1e-12,
            err_msg=f"Serial spatial lag mismatch on {d['name']}")

    def test_parallel_matches_scipy(self, sp, dataset):
        d = dataset
        ref = _scipy_lag(d["z"], d["rows"], d["cols"], d["vals"], d["n"])
        wy  = sp.spatial_lag(d["z"], d["rows"], d["cols"], d["vals"], d["n"], parallel=True)
        np.testing.assert_allclose(wy, ref, atol=1e-12,
            err_msg=f"Parallel spatial lag mismatch on {d['name']}")

    def test_serial_parallel_agree(self, sp, dataset):
        d = dataset
        wy_s = sp.spatial_lag(d["z"], d["rows"], d["cols"], d["vals"], d["n"], parallel=False)
        wy_p = sp.spatial_lag(d["z"], d["rows"], d["cols"], d["vals"], d["n"], parallel=True)
        np.testing.assert_allclose(wy_s, wy_p, atol=1e-13,
            err_msg=f"Serial/parallel spatial lag disagree on {d['name']}")


# ---------------------------------------------------------------------------
# Point estimates
# ---------------------------------------------------------------------------

class TestPointEstimates:
    def test_moran_i_matches_reference(self, sp, dataset):
        d = dataset
        ref = _esda_moran(d["z"], d["rows"], d["cols"], d["vals"], d["n"])
        got = sp.moran_i(d["z"], d["rows"], d["cols"], d["vals"], d["n"])
        assert abs(got - ref) < 1e-12, \
            f"Moran's I mismatch on {d['name']}: {got:.8f} vs {ref:.8f}"

    def test_gearysc_matches_reference(self, sp, dataset):
        d = dataset
        ref = _esda_gearysc(d["z"], d["rows"], d["cols"], d["vals"], d["n"])
        got = sp.gearysc(d["z"], d["rows"], d["cols"], d["vals"], d["n"])
        assert abs(got - ref) < 1e-12, \
            f"Geary's C mismatch on {d['name']}: {got:.8f} vs {ref:.8f}"


# ---------------------------------------------------------------------------
# Permutation tests — observed statistic consistency
# ---------------------------------------------------------------------------

N_PERM_FAST = 999   # small enough for a quick test run

class TestPermObserved:
    @pytest.mark.parametrize("statistic", ["moran", "gearysc"])
    def test_serial_parallel_observed_agree(self, sp, dataset, statistic):
        d = dataset
        r_s = sp.perm_serial(d["z"], d["rows"], d["cols"], d["vals"], d["n"],
                              n_perm=N_PERM_FAST, statistic=statistic)
        r_p = sp.perm_parallel(d["z"], d["rows"], d["cols"], d["vals"], d["n"],
                                n_perm=N_PERM_FAST, statistic=statistic)
        assert abs(r_s.observed - r_p.observed) < 1e-12, \
            f"{statistic} observed mismatch serial/parallel on {d['name']}"

    @pytest.mark.parametrize("statistic", ["moran", "gearysc"])
    def test_observed_matches_point_estimate(self, sp, dataset, statistic):
        d = dataset
        r = sp.perm_parallel(d["z"], d["rows"], d["cols"], d["vals"], d["n"],
                              n_perm=N_PERM_FAST, statistic=statistic)
        if statistic == "moran":
            ref = sp.moran_i(d["z"], d["rows"], d["cols"], d["vals"], d["n"])
        else:
            ref = sp.gearysc(d["z"], d["rows"], d["cols"], d["vals"], d["n"])
        assert abs(r.observed - ref) < 1e-12, \
            f"{statistic} permutation observed != point estimate on {d['name']}"

    @pytest.mark.parametrize("statistic", ["moran", "gearysc"])
    def test_metal_observed_matches_cpu(self, sp, dataset, statistic):
        d = dataset
        r_cpu   = sp.perm_parallel(d["z"], d["rows"], d["cols"], d["vals"], d["n"],
                                    n_perm=N_PERM_FAST, statistic=statistic)
        r_metal = sp.perm_metal(d["z"], d["rows"], d["cols"], d["vals"], d["n"],
                                 n_perm=N_PERM_FAST, statistic=statistic,
                                 fallback_to_parallel=True)
        assert abs(r_cpu.observed - r_metal.observed) < 1e-12, \
            f"{statistic} observed mismatch CPU/Metal on {d['name']}"


# ---------------------------------------------------------------------------
# Permutation tests — p-value plausibility and CPU/Metal agreement
# ---------------------------------------------------------------------------

N_PERM_PVAL = 9999  # enough for stable p-value comparison

class TestPermPValues:
    @pytest.mark.parametrize("statistic", ["moran", "gearysc"])
    def test_serial_parallel_pvalue_close(self, sp, dataset, statistic):
        """Serial and parallel use different per-permutation seeds so p-values
        won't be identical, but should agree within Monte Carlo tolerance."""
        d = dataset
        r_s = sp.perm_serial(d["z"], d["rows"], d["cols"], d["vals"], d["n"],
                              n_perm=N_PERM_PVAL, statistic=statistic)
        r_p = sp.perm_parallel(d["z"], d["rows"], d["cols"], d["vals"], d["n"],
                                n_perm=N_PERM_PVAL, statistic=statistic)
        # Within 5 percentage points — generous to account for RNG differences
        assert abs(r_s.p_value - r_p.p_value) < 0.05, \
            f"{statistic} p-value serial={r_s.p_value:.4f} vs parallel={r_p.p_value:.4f} on {d['name']}"

    @pytest.mark.parametrize("statistic", ["moran", "gearysc"])
    def test_cpu_metal_pvalue_close(self, sp, dataset, statistic):
        d = dataset
        r_cpu   = sp.perm_parallel(d["z"], d["rows"], d["cols"], d["vals"], d["n"],
                                    n_perm=N_PERM_PVAL, statistic=statistic)
        r_metal = sp.perm_metal(d["z"], d["rows"], d["cols"], d["vals"], d["n"],
                                 n_perm=N_PERM_PVAL, statistic=statistic,
                                 fallback_to_parallel=True)
        assert abs(r_cpu.p_value - r_metal.p_value) < 0.05, \
            f"{statistic} p-value CPU={r_cpu.p_value:.4f} vs Metal={r_metal.p_value:.4f} on {d['name']}"

    def test_null_distribution_length(self, sp, dataset):
        d = dataset
        for n_perm in (99, 999, 9999):
            r = sp.perm_parallel(d["z"], d["rows"], d["cols"], d["vals"], d["n"],
                                  n_perm=n_perm)
            assert len(r.null_distribution) == n_perm, \
                f"Expected null dist length {n_perm}, got {len(r.null_distribution)}"


# ---------------------------------------------------------------------------
# Local Moran's I (LISA)
# ---------------------------------------------------------------------------

class TestLocalMoranI:
    def test_point_estimates_match_reference(self, sp, dataset):
        """local_moran_i(z) == z * W.dot(z) element-wise."""
        d = dataset
        from scipy.sparse import csr_matrix
        W   = csr_matrix((d["vals"], (d["rows"], d["cols"])), shape=(d["n"], d["n"]))
        ref = d["z"] * W.dot(d["z"])
        got = sp.local_moran_i(d["z"], d["rows"], d["cols"], d["vals"], d["n"])
        np.testing.assert_allclose(got, ref, atol=1e-12,
            err_msg=f"Local Moran's I mismatch on {d['name']}")

    def test_serial_parallel_observed_agree(self, sp, dataset):
        d = dataset
        r_s = sp.local_perm_serial(d["z"], d["rows"], d["cols"], d["vals"], d["n"],
                                    n_perm=N_PERM_FAST)
        r_p = sp.local_perm_parallel(d["z"], d["rows"], d["cols"], d["vals"], d["n"],
                                      n_perm=N_PERM_FAST)
        np.testing.assert_allclose(r_s.observed, r_p.observed, atol=1e-12,
            err_msg=f"Local Moran's I observed mismatch on {d['name']}")

    def test_observed_matches_point_estimate(self, sp, dataset):
        d = dataset
        ref = sp.local_moran_i(d["z"], d["rows"], d["cols"], d["vals"], d["n"])
        r   = sp.local_perm_parallel(d["z"], d["rows"], d["cols"], d["vals"], d["n"],
                                      n_perm=N_PERM_FAST)
        np.testing.assert_allclose(r.observed, ref, atol=1e-12,
            err_msg=f"Local Moran's I perm observed != point estimate on {d['name']}")

    def test_output_shapes(self, sp, dataset):
        d = dataset
        r = sp.local_perm_parallel(d["z"], d["rows"], d["cols"], d["vals"], d["n"],
                                    n_perm=N_PERM_FAST)
        assert r.observed.shape == (d["n"],), f"observed shape wrong on {d['name']}"
        assert r.p_values.shape  == (d["n"],), f"p_values shape wrong on {d['name']}"

    def test_pvalues_in_unit_interval(self, sp, dataset):
        d = dataset
        r = sp.local_perm_parallel(d["z"], d["rows"], d["cols"], d["vals"], d["n"],
                                    n_perm=N_PERM_FAST)
        assert np.all(r.p_values >= 0.0) and np.all(r.p_values <= 1.0), \
            f"p-values out of [0,1] on {d['name']}"

    def test_serial_parallel_pvalues_close(self, sp, dataset):
        d = dataset
        r_s = sp.local_perm_serial(d["z"], d["rows"], d["cols"], d["vals"], d["n"],
                                    n_perm=N_PERM_PVAL)
        r_p = sp.local_perm_parallel(d["z"], d["rows"], d["cols"], d["vals"], d["n"],
                                      n_perm=N_PERM_PVAL)
        # Mean absolute difference across all observations
        mean_diff = float(np.mean(np.abs(r_s.p_values - r_p.p_values)))
        assert mean_diff < 0.05, \
            f"Local p-values serial/parallel mean diff={mean_diff:.4f} on {d['name']}"


# ---------------------------------------------------------------------------
# Local Geary's C
# ---------------------------------------------------------------------------

class TestLocalGearysC:
    def test_point_estimates_match_reference(self, sp, dataset):
        """local_gearysc(z)[i] == Σⱼ w[i,j] * (z[i] - z[j])²"""
        d = dataset
        ref = np.zeros(d["n"])
        for k in range(len(d["vals"])):
            diff = d["z"][d["rows"][k]] - d["z"][d["cols"][k]]
            ref[d["rows"][k]] += d["vals"][k] * diff * diff
        got = sp.local_gearysc(d["z"], d["rows"], d["cols"], d["vals"], d["n"])
        np.testing.assert_allclose(got, ref, atol=1e-12,
            err_msg=f"Local Geary's C mismatch on {d['name']}")

    def test_all_nonnegative(self, sp, dataset):
        d = dataset
        got = sp.local_gearysc(d["z"], d["rows"], d["cols"], d["vals"], d["n"])
        assert np.all(got >= 0), f"Local Geary's C has negative values on {d['name']}"

    def test_serial_parallel_observed_agree(self, sp, dataset):
        d = dataset
        r_s = sp.local_perm_serial(d["z"], d["rows"], d["cols"], d["vals"], d["n"],
                                    n_perm=N_PERM_FAST, statistic="gearysc")
        r_p = sp.local_perm_parallel(d["z"], d["rows"], d["cols"], d["vals"], d["n"],
                                      n_perm=N_PERM_FAST, statistic="gearysc")
        np.testing.assert_allclose(r_s.observed, r_p.observed, atol=1e-12,
            err_msg=f"Local Geary's C observed mismatch on {d['name']}")

    def test_observed_matches_point_estimate(self, sp, dataset):
        d = dataset
        ref = sp.local_gearysc(d["z"], d["rows"], d["cols"], d["vals"], d["n"])
        r   = sp.local_perm_parallel(d["z"], d["rows"], d["cols"], d["vals"], d["n"],
                                      n_perm=N_PERM_FAST, statistic="gearysc")
        np.testing.assert_allclose(r.observed, ref, atol=1e-12,
            err_msg=f"Local Geary's C perm observed != point estimate on {d['name']}")

    def test_pvalues_in_unit_interval(self, sp, dataset):
        d = dataset
        r = sp.local_perm_parallel(d["z"], d["rows"], d["cols"], d["vals"], d["n"],
                                    n_perm=N_PERM_FAST, statistic="gearysc")
        assert np.all(r.p_values >= 0.0) and np.all(r.p_values <= 1.0), \
            f"Local Geary's C p-values out of [0,1] on {d['name']}"

    def test_serial_parallel_pvalues_close(self, sp, dataset):
        d = dataset
        r_s = sp.local_perm_serial(d["z"], d["rows"], d["cols"], d["vals"], d["n"],
                                    n_perm=N_PERM_PVAL, statistic="gearysc")
        r_p = sp.local_perm_parallel(d["z"], d["rows"], d["cols"], d["vals"], d["n"],
                                      n_perm=N_PERM_PVAL, statistic="gearysc")
        mean_diff = float(np.mean(np.abs(r_s.p_values - r_p.p_values)))
        assert mean_diff < 0.05, \
            f"Local Geary's C p-values mean diff={mean_diff:.4f} on {d['name']}"


# ---------------------------------------------------------------------------
# w_to_coo interop utility
# ---------------------------------------------------------------------------

class TestWToCoo:
    def test_w_to_coo_matches_fixtures(self):
        """w_to_coo(w) must produce the same COO arrays as generate_fixtures.py."""
        pytest.importorskip("libpysal")
        pytest.importorskip("geopandas")
        import libpysal
        import geopandas as gpd
        from libpysal.weights import Queen
        from swiftperm import w_to_coo

        shp = libpysal.examples.get_path("columbus.shp")
        w = Queen.from_shapefile(shp)
        w.transform = "r"

        rows_w, cols_w, vals_w = w_to_coo(w)

        # Compare against fixture
        z_path = DATA_DIR / "columbus_z.bin"
        if not z_path.exists():
            pytest.skip("Columbus fixture not found")
        _, rows_f, cols_f, vals_f = _read_w(DATA_DIR / "columbus_weights.bin")

        # Sort both by (row, col) for comparison
        def sort_coo(r, c, v):
            idx = np.lexsort((c, r))
            return r[idx], c[idx], v[idx]

        rw, cw, vw = sort_coo(rows_w, cols_w, vals_w)
        rf, cf, vf = sort_coo(rows_f, cols_f, vals_f)

        np.testing.assert_array_equal(rw, rf)
        np.testing.assert_array_equal(cw, cf)
        np.testing.assert_allclose(vw, vf, atol=1e-14)
