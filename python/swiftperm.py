"""
swiftperm.py
Python ctypes binding for the SwiftGeoLib dynamic library.

Exposes three permutation backends (serial, parallel, Metal) for four statistics:
Moran's I, Geary's C, Getis-Ord G (CPU only), and join count (CPU only).

All numpy inputs are copied to contiguous float64/int32 arrays before passing to C.
The null distribution is written directly into a pre-allocated numpy array;
no Swift heap allocation or free function is needed.

Usage:
    from swiftperm import SwiftPerm
    sp = SwiftPerm()
    result = sp.perm_parallel(z, rows, cols, vals, n, statistic="moran")
    result = sp.perm_parallel(z, rows, cols, vals, n, statistic="gearysc")
    result = sp.perm_metal(z, rows, cols, vals, n, statistic="gearysc")
    print(result.observed, result.p_value)

Statistics:
    "moran"     Moran's I — standardized z, GPU supported
    "gearysc"   Geary's C — standardized z, GPU supported
    "getisordg" Getis-Ord G — positive z, CPU only
    "joincount" Join count — binary z ∈ {0,1}, CPU only

Environment:
    SWIFTGEO_DYLIB   Override dylib path (default: ../SwiftGeo/.build/release/libSwiftGeoLib.dylib)
"""

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_DEFAULT_DYLIB = (
    Path(__file__).parent.parent / "SwiftGeo" / ".build" / "release" / "libSwiftGeoLib.dylib"
)

_dbl_p = ctypes.POINTER(ctypes.c_double)
_i32_p = ctypes.POINTER(ctypes.c_int32)

_WEIGHT_ARGS = [
    _dbl_p, ctypes.c_int32,   # z_ptr, z_len
    _i32_p, _i32_p, _dbl_p,   # rows_ptr, cols_ptr, vals_ptr
    ctypes.c_int32,            # nnz
    ctypes.c_int32,            # n
]
_OUT_ARGS = [
    _dbl_p,                    # out_null
    _dbl_p,                    # out_observed
    _dbl_p,                    # out_p_value
    _dbl_p,                    # out_elapsed
    _i32_p,                    # out_nthreads
]

# Maps statistic name → C symbol suffix
_STAT_SUFFIX = {
    "moran":     "",
    "gearysc":   "_gearysc",
    "getisordg": "_getisordg",
    "joincount":  "_joincount",
}
# Statistics with Metal GPU paths
_METAL_STATS = {"moran", "gearysc"}

_lib = None


def w_to_coo(w):
    """Extract contiguous int32/float64 COO arrays from a libpysal W object.

    Returns (rows, cols, vals) as numpy arrays ready to pass to SwiftPerm methods.
    This is the interop bridge between libpysal and the Swift backend.
    """
    id_order  = w.id_order
    id_to_idx = {id_: i for i, id_ in enumerate(id_order)}
    rows, cols, vals = [], [], []
    for focal_id, neighbors in w.neighbors.items():
        i = id_to_idx[focal_id]
        for neighbor_id, wval in zip(neighbors, w.weights[focal_id]):
            rows.append(i)
            cols.append(id_to_idx[neighbor_id])
            vals.append(wval)
    return (np.ascontiguousarray(rows, dtype=np.int32),
            np.ascontiguousarray(cols, dtype=np.int32),
            np.ascontiguousarray(vals, dtype=np.float64))


def _load_lib(dylib_path=None):
    global _lib
    if _lib is not None:
        return _lib

    path = dylib_path or os.environ.get("SWIFTGEO_DYLIB") or str(_DEFAULT_DYLIB)
    lib = ctypes.CDLL(str(path))

    # Local Moran's I point estimates — writes n doubles
    fn = lib.swiftgeo_local_moran_i
    fn.restype  = None
    fn.argtypes = _WEIGHT_ARGS + [_dbl_p]

    # Local permutation tests — writes n observed + n p-values + elapsed + nthreads
    for name in ("swiftgeo_local_perm_serial", "swiftgeo_local_perm_parallel"):
        fn = getattr(lib, name)
        fn.restype  = None
        fn.argtypes = _WEIGHT_ARGS + [ctypes.c_int32, ctypes.c_uint64,
                                       _dbl_p, _dbl_p, _dbl_p, _i32_p]

    # Spatial lag (serial + parallel) — output buffer instead of scalar
    for name in ("swiftgeo_spatial_lag", "swiftgeo_spatial_lag_parallel"):
        fn = getattr(lib, name)
        fn.restype  = None
        fn.argtypes = _WEIGHT_ARGS + [_dbl_p]   # out_wy

    # Point estimate functions
    for name, restype in [("swiftgeo_moran_i", ctypes.c_double),
                           ("swiftgeo_gearysc",  ctypes.c_double)]:
        fn = getattr(lib, name)
        fn.restype  = restype
        fn.argtypes = _WEIGHT_ARGS

    # CPU perm functions (serial + parallel, all statistics)
    for suffix in ("", "_gearysc", "_getisordg", "_joincount"):
        for variant in ("serial", "parallel"):
            fn = getattr(lib, f"swiftgeo_perm_{variant}{suffix}")
            fn.restype  = None
            fn.argtypes = _WEIGHT_ARGS + [ctypes.c_int32, ctypes.c_uint64] + _OUT_ARGS

    # Metal perm functions (Moran's I and Geary's C only — seed is UInt32)
    for suffix in ("", "_gearysc"):
        fn = getattr(lib, f"swiftgeo_perm_metal{suffix}")
        fn.restype  = ctypes.c_int32
        fn.argtypes = _WEIGHT_ARGS + [ctypes.c_int32, ctypes.c_uint32] + _OUT_ARGS

    _lib = lib
    return lib


@dataclass
class LocalPermResult:
    observed: np.ndarray        # shape (n,), float64 — local Moran's I values
    p_values: np.ndarray        # shape (n,), float64 — two-sided p-values
    elapsed_seconds: float
    n_threads: int


@dataclass
class PermResult:
    observed: float
    null_distribution: np.ndarray  # shape (n_perm,), float64
    p_value: float
    elapsed_seconds: float
    n_threads: int


class SwiftPerm:
    def __init__(self, dylib_path=None):
        self._lib = _load_lib(dylib_path)

    @staticmethod
    def _prep(z, rows, cols, vals):
        z_c    = np.ascontiguousarray(z,    dtype=np.float64)
        rows_c = np.ascontiguousarray(rows, dtype=np.int32)
        cols_c = np.ascontiguousarray(cols, dtype=np.int32)
        vals_c = np.ascontiguousarray(vals, dtype=np.float64)
        return z_c, rows_c, cols_c, vals_c

    @staticmethod
    def _weight_args(z_c, rows_c, cols_c, vals_c, n):
        return (
            z_c.ctypes.data_as(_dbl_p),    ctypes.c_int32(len(z_c)),
            rows_c.ctypes.data_as(_i32_p), cols_c.ctypes.data_as(_i32_p),
            vals_c.ctypes.data_as(_dbl_p),
            ctypes.c_int32(len(vals_c)),   ctypes.c_int32(n),
        )

    def _call(self, fn, z, rows, cols, vals, n, n_perm, seed_arg):
        z_c, rows_c, cols_c, vals_c = self._prep(z, rows, cols, vals)
        null = np.empty(n_perm, dtype=np.float64)
        obs, pv, el, nt = (ctypes.c_double(), ctypes.c_double(),
                           ctypes.c_double(), ctypes.c_int32())
        fn(
            *self._weight_args(z_c, rows_c, cols_c, vals_c, n),
            ctypes.c_int32(n_perm), seed_arg,
            null.ctypes.data_as(_dbl_p),
            ctypes.byref(obs), ctypes.byref(pv),
            ctypes.byref(el),  ctypes.byref(nt),
        )
        return PermResult(obs.value, null, pv.value, el.value, nt.value)

    def local_moran_i(self, z, rows, cols, vals, n) -> np.ndarray:
        """Point estimates: returns array of n local Moran's I values."""
        z_c, rows_c, cols_c, vals_c = self._prep(z, rows, cols, vals)
        out = np.empty(n, dtype=np.float64)
        self._lib.swiftgeo_local_moran_i(
            *self._weight_args(z_c, rows_c, cols_c, vals_c, n),
            out.ctypes.data_as(_dbl_p),
        )
        return out

    def _call_local(self, fn, z, rows, cols, vals, n, n_perm, seed):
        z_c, rows_c, cols_c, vals_c = self._prep(z, rows, cols, vals)
        observed = np.empty(n, dtype=np.float64)
        p_values = np.empty(n, dtype=np.float64)
        el, nt   = ctypes.c_double(), ctypes.c_int32()
        fn(
            *self._weight_args(z_c, rows_c, cols_c, vals_c, n),
            ctypes.c_int32(n_perm), ctypes.c_uint64(seed),
            observed.ctypes.data_as(_dbl_p),
            p_values.ctypes.data_as(_dbl_p),
            ctypes.byref(el), ctypes.byref(nt),
        )
        return LocalPermResult(observed, p_values, el.value, nt.value)

    def local_perm_serial(self, z, rows, cols, vals, n,
                          n_perm=9999, seed=12345) -> LocalPermResult:
        return self._call_local(self._lib.swiftgeo_local_perm_serial,
                                z, rows, cols, vals, n, n_perm, seed)

    def local_perm_parallel(self, z, rows, cols, vals, n,
                            n_perm=9999, seed=12345) -> LocalPermResult:
        return self._call_local(self._lib.swiftgeo_local_perm_parallel,
                                z, rows, cols, vals, n, n_perm, seed)

    def spatial_lag(self, y, rows, cols, vals, n, parallel=True) -> np.ndarray:
        """Compute the spatial lag W·y.

        Parameters match the permutation methods. For libpysal W objects use
        w_to_coo(w) to extract rows/cols/vals first.

        Falls back to scipy sparse multiply if the dylib is unavailable — call
        this via the module-level spatial_lag() convenience function for that.
        """
        y_c, rows_c, cols_c, vals_c = self._prep(y, rows, cols, vals)
        wy = np.empty(n, dtype=np.float64)
        fn = (self._lib.swiftgeo_spatial_lag_parallel if parallel
              else self._lib.swiftgeo_spatial_lag)
        fn(
            *self._weight_args(y_c, rows_c, cols_c, vals_c, n),
            wy.ctypes.data_as(_dbl_p),
        )
        return wy

    def moran_i(self, z, rows, cols, vals, n) -> float:
        z_c, rows_c, cols_c, vals_c = self._prep(z, rows, cols, vals)
        return self._lib.swiftgeo_moran_i(
            *self._weight_args(z_c, rows_c, cols_c, vals_c, n)
        )

    def gearysc(self, z, rows, cols, vals, n) -> float:
        z_c, rows_c, cols_c, vals_c = self._prep(z, rows, cols, vals)
        return self._lib.swiftgeo_gearysc(
            *self._weight_args(z_c, rows_c, cols_c, vals_c, n)
        )

    def perm_serial(self, z, rows, cols, vals, n, n_perm=99999, seed=12345,
                    statistic="moran") -> PermResult:
        suffix = _STAT_SUFFIX[statistic]
        fn = getattr(self._lib, f"swiftgeo_perm_serial{suffix}")
        return self._call(fn, z, rows, cols, vals, n, n_perm, ctypes.c_uint64(seed))

    def perm_parallel(self, z, rows, cols, vals, n, n_perm=99999, seed=12345,
                      statistic="moran") -> PermResult:
        suffix = _STAT_SUFFIX[statistic]
        fn = getattr(self._lib, f"swiftgeo_perm_parallel{suffix}")
        return self._call(fn, z, rows, cols, vals, n, n_perm, ctypes.c_uint64(seed))

    def perm_metal(self, z, rows, cols, vals, n, n_perm=99999, seed=12345,
                   statistic="moran", fallback_to_parallel=True) -> PermResult:
        if statistic not in _METAL_STATS:
            return self.perm_parallel(z, rows, cols, vals, n, n_perm, seed,
                                      statistic=statistic)
        suffix = _STAT_SUFFIX[statistic]
        fn     = getattr(self._lib, f"swiftgeo_perm_metal{suffix}")
        z_c, rows_c, cols_c, vals_c = self._prep(z, rows, cols, vals)
        null = np.empty(n_perm, dtype=np.float64)
        obs, pv, el, nt = (ctypes.c_double(), ctypes.c_double(),
                           ctypes.c_double(), ctypes.c_int32())
        # Metal seed is UInt32; seeds > 2^32 are silently truncated
        ok = fn(
            *self._weight_args(z_c, rows_c, cols_c, vals_c, n),
            ctypes.c_int32(n_perm), ctypes.c_uint32(seed & 0xFFFFFFFF),
            null.ctypes.data_as(_dbl_p),
            ctypes.byref(obs), ctypes.byref(pv),
            ctypes.byref(el),  ctypes.byref(nt),
        )
        if ok == 0:
            if fallback_to_parallel:
                return self.perm_parallel(z, rows, cols, vals, n, n_perm, seed,
                                          statistic=statistic)
            raise RuntimeError("Metal unavailable and fallback disabled")
        return PermResult(obs.value, null, pv.value, el.value, nt.value)
