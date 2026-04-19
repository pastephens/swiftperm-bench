// bridge.swift
// C-compatible bridge for Python ctypes binding.
// All functions take raw pointers; Swift reconstructs value types internally.
// The null distribution is written into a caller-owned buffer (no Swift heap allocation).

import SwiftGeo

private func makeWeights(
    rows_ptr: UnsafePointer<Int32>, cols_ptr: UnsafePointer<Int32>,
    vals_ptr: UnsafePointer<Double>, nnz: Int32, n: Int32
) -> SparseWeights {
    let k = Int(nnz)
    return SparseWeights(
        rows:   Array(UnsafeBufferPointer(start: rows_ptr, count: k)),
        cols:   Array(UnsafeBufferPointer(start: cols_ptr, count: k)),
        values: Array(UnsafeBufferPointer(start: vals_ptr, count: k)),
        n: Int(n)
    )
}

private func writeResult(
    _ r: PermutationResult,
    out_null:     UnsafeMutablePointer<Double>,
    out_observed: UnsafeMutablePointer<Double>,
    out_p_value:  UnsafeMutablePointer<Double>,
    out_elapsed:  UnsafeMutablePointer<Double>,
    out_nthreads: UnsafeMutablePointer<Int32>
) {
    r.nullDistribution.withUnsafeBufferPointer {
        out_null.initialize(from: $0.baseAddress!, count: $0.count)
    }
    out_observed.pointee = r.observed
    out_p_value.pointee  = r.pValueTwoSided
    out_elapsed.pointee  = r.elapsedSeconds
    out_nthreads.pointee = Int32(r.nThreads)
}

@_cdecl("swiftgeo_moran_i")
public func swiftgeo_moran_i(
    z_ptr:    UnsafePointer<Double>, z_len: Int32,
    rows_ptr: UnsafePointer<Int32>,
    cols_ptr: UnsafePointer<Int32>,
    vals_ptr: UnsafePointer<Double>, nnz: Int32, n: Int32
) -> Double {
    let z = Array(UnsafeBufferPointer(start: z_ptr, count: Int(z_len)))
    return moranI(z: z, weights: makeWeights(rows_ptr: rows_ptr, cols_ptr: cols_ptr,
                                              vals_ptr: vals_ptr, nnz: nnz, n: n))
}

@_cdecl("swiftgeo_perm_serial")
public func swiftgeo_perm_serial(
    z_ptr:    UnsafePointer<Double>, z_len: Int32,
    rows_ptr: UnsafePointer<Int32>,
    cols_ptr: UnsafePointer<Int32>,
    vals_ptr: UnsafePointer<Double>, nnz: Int32, n: Int32,
    n_perm: Int32, seed: UInt64,
    out_null:     UnsafeMutablePointer<Double>,
    out_observed: UnsafeMutablePointer<Double>,
    out_p_value:  UnsafeMutablePointer<Double>,
    out_elapsed:  UnsafeMutablePointer<Double>,
    out_nthreads: UnsafeMutablePointer<Int32>
) {
    let z = Array(UnsafeBufferPointer(start: z_ptr, count: Int(z_len)))
    let r = moranPermutationTestSerial(
        z: z,
        weights: makeWeights(rows_ptr: rows_ptr, cols_ptr: cols_ptr, vals_ptr: vals_ptr, nnz: nnz, n: n),
        nPermutations: Int(n_perm), seed: seed)
    writeResult(r, out_null: out_null, out_observed: out_observed,
                out_p_value: out_p_value, out_elapsed: out_elapsed, out_nthreads: out_nthreads)
}

@_cdecl("swiftgeo_perm_parallel")
public func swiftgeo_perm_parallel(
    z_ptr:    UnsafePointer<Double>, z_len: Int32,
    rows_ptr: UnsafePointer<Int32>,
    cols_ptr: UnsafePointer<Int32>,
    vals_ptr: UnsafePointer<Double>, nnz: Int32, n: Int32,
    n_perm: Int32, seed: UInt64,
    out_null:     UnsafeMutablePointer<Double>,
    out_observed: UnsafeMutablePointer<Double>,
    out_p_value:  UnsafeMutablePointer<Double>,
    out_elapsed:  UnsafeMutablePointer<Double>,
    out_nthreads: UnsafeMutablePointer<Int32>
) {
    let z = Array(UnsafeBufferPointer(start: z_ptr, count: Int(z_len)))
    let r = moranPermutationTest(
        z: z,
        weights: makeWeights(rows_ptr: rows_ptr, cols_ptr: cols_ptr, vals_ptr: vals_ptr, nnz: nnz, n: n),
        nPermutations: Int(n_perm), seed: seed)
    writeResult(r, out_null: out_null, out_observed: out_observed,
                out_p_value: out_p_value, out_elapsed: out_elapsed, out_nthreads: out_nthreads)
}

// Returns 1 on success, 0 if Metal is unavailable.
// seed is UInt32 (matches MetalPermutation.swift) — seeds > 2^32 will be truncated on the Python side.
@_cdecl("swiftgeo_perm_metal")
public func swiftgeo_perm_metal(
    z_ptr:    UnsafePointer<Double>, z_len: Int32,
    rows_ptr: UnsafePointer<Int32>,
    cols_ptr: UnsafePointer<Int32>,
    vals_ptr: UnsafePointer<Double>, nnz: Int32, n: Int32,
    n_perm: Int32, seed: UInt32,
    out_null:     UnsafeMutablePointer<Double>,
    out_observed: UnsafeMutablePointer<Double>,
    out_p_value:  UnsafeMutablePointer<Double>,
    out_elapsed:  UnsafeMutablePointer<Double>,
    out_nthreads: UnsafeMutablePointer<Int32>
) -> Int32 {
    let z = Array(UnsafeBufferPointer(start: z_ptr, count: Int(z_len)))
    guard let r = moranPermutationTestMetal(
        z: z,
        weights: makeWeights(rows_ptr: rows_ptr, cols_ptr: cols_ptr, vals_ptr: vals_ptr, nnz: nnz, n: n),
        nPermutations: Int(n_perm), seed: seed)
    else { return 0 }
    writeResult(r, out_null: out_null, out_observed: out_observed,
                out_p_value: out_p_value, out_elapsed: out_elapsed, out_nthreads: out_nthreads)
    return 1
}
