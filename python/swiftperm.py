"""
swiftperm.py
Python ctypes binding for the SwiftGeoLib dynamic library.

Exposes three permutation backends (serial, parallel, Metal) and a point
estimate of Moran's I. All numpy inputs are copied to contiguous float64/int32
arrays before passing to C — slices and other dtypes are handled automatically.

The null distribution is written directly into a pre-allocated numpy array;
no Swift heap allocation or free function is needed.

Usage:
    from swiftperm import SwiftPerm
    sp = SwiftPerm()                             # auto-discovers dylib
    result = sp.perm_parallel(z, rows, cols, vals, n)
    print(result.observed, result.p_value)

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

_lib = None


def _load_lib(dylib_path=None):
    global _lib
    if _lib is not None:
        return _lib

    path = dylib_path or os.environ.get("SWIFTGEO_DYLIB") or str(_DEFAULT_DYLIB)
    lib = ctypes.CDLL(str(path))

    lib.swiftgeo_moran_i.restype  = ctypes.c_double
    lib.swiftgeo_moran_i.argtypes = _WEIGHT_ARGS

    for name in ("swiftgeo_perm_serial", "swiftgeo_perm_parallel"):
        fn = getattr(lib, name)
        fn.restype  = None
        fn.argtypes = _WEIGHT_ARGS + [ctypes.c_int32, ctypes.c_uint64] + _OUT_ARGS

    lib.swiftgeo_perm_metal.restype  = ctypes.c_int32
    # Metal seed is UInt32 (not UInt64) — matches MetalPermutation.swift
    lib.swiftgeo_perm_metal.argtypes = _WEIGHT_ARGS + [ctypes.c_int32, ctypes.c_uint32] + _OUT_ARGS

    _lib = lib
    return lib


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

    def moran_i(self, z, rows, cols, vals, n) -> float:
        z_c, rows_c, cols_c, vals_c = self._prep(z, rows, cols, vals)
        return self._lib.swiftgeo_moran_i(
            *self._weight_args(z_c, rows_c, cols_c, vals_c, n)
        )

    def perm_serial(self, z, rows, cols, vals, n, n_perm=99999, seed=12345) -> PermResult:
        return self._call(self._lib.swiftgeo_perm_serial,
                          z, rows, cols, vals, n, n_perm, ctypes.c_uint64(seed))

    def perm_parallel(self, z, rows, cols, vals, n, n_perm=99999, seed=12345) -> PermResult:
        return self._call(self._lib.swiftgeo_perm_parallel,
                          z, rows, cols, vals, n, n_perm, ctypes.c_uint64(seed))

    def perm_metal(self, z, rows, cols, vals, n, n_perm=99999, seed=12345,
                   fallback_to_parallel=True) -> PermResult:
        z_c, rows_c, cols_c, vals_c = self._prep(z, rows, cols, vals)
        null = np.empty(n_perm, dtype=np.float64)
        obs, pv, el, nt = (ctypes.c_double(), ctypes.c_double(),
                           ctypes.c_double(), ctypes.c_int32())
        # Seed truncated to UInt32; CPU paths use full UInt64
        ok = self._lib.swiftgeo_perm_metal(
            *self._weight_args(z_c, rows_c, cols_c, vals_c, n),
            ctypes.c_int32(n_perm), ctypes.c_uint32(seed & 0xFFFFFFFF),
            null.ctypes.data_as(_dbl_p),
            ctypes.byref(obs), ctypes.byref(pv),
            ctypes.byref(el),  ctypes.byref(nt),
        )
        if ok == 0:
            if fallback_to_parallel:
                return self.perm_parallel(z, rows, cols, vals, n, n_perm, seed)
            raise RuntimeError("Metal unavailable and fallback disabled")
        return PermResult(obs.value, null, pv.value, el.value, nt.value)
