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

// MARK: - Local Moran's I

@_cdecl("swiftgeo_local_moran_i")
public func swiftgeo_local_moran_i(
    z_ptr:    UnsafePointer<Double>, z_len: Int32,
    rows_ptr: UnsafePointer<Int32>,
    cols_ptr: UnsafePointer<Int32>,
    vals_ptr: UnsafePointer<Double>, nnz: Int32, n: Int32,
    out_local: UnsafeMutablePointer<Double>
) {
    let z      = Array(UnsafeBufferPointer(start: z_ptr, count: Int(z_len)))
    let local  = localMoranI(z: z, weights: makeWeights(rows_ptr: rows_ptr, cols_ptr: cols_ptr,
                                                         vals_ptr: vals_ptr, nnz: nnz, n: n))
    out_local.initialize(from: local, count: local.count)
}

private func writeLocalResult(
    _ r: LocalPermutationResult,
    out_observed: UnsafeMutablePointer<Double>,
    out_p_values: UnsafeMutablePointer<Double>,
    out_elapsed:  UnsafeMutablePointer<Double>,
    out_nthreads: UnsafeMutablePointer<Int32>
) {
    r.observed.withUnsafeBufferPointer {
        out_observed.initialize(from: $0.baseAddress!, count: $0.count)
    }
    r.pValues.withUnsafeBufferPointer {
        out_p_values.initialize(from: $0.baseAddress!, count: $0.count)
    }
    out_elapsed.pointee  = r.elapsedSeconds
    out_nthreads.pointee = Int32(r.nThreads)
}

@_cdecl("swiftgeo_local_perm_serial")
public func swiftgeo_local_perm_serial(
    z_ptr:    UnsafePointer<Double>, z_len: Int32,
    rows_ptr: UnsafePointer<Int32>,
    cols_ptr: UnsafePointer<Int32>,
    vals_ptr: UnsafePointer<Double>, nnz: Int32, n: Int32,
    n_perm: Int32, seed: UInt64,
    out_observed: UnsafeMutablePointer<Double>,
    out_p_values: UnsafeMutablePointer<Double>,
    out_elapsed:  UnsafeMutablePointer<Double>,
    out_nthreads: UnsafeMutablePointer<Int32>
) {
    let z = Array(UnsafeBufferPointer(start: z_ptr, count: Int(z_len)))
    let r = localPermutationTestSerial(
        z: z,
        weights: makeWeights(rows_ptr: rows_ptr, cols_ptr: cols_ptr, vals_ptr: vals_ptr, nnz: nnz, n: n),
        nPermutations: Int(n_perm), seed: seed)
    writeLocalResult(r, out_observed: out_observed, out_p_values: out_p_values,
                     out_elapsed: out_elapsed, out_nthreads: out_nthreads)
}

@_cdecl("swiftgeo_local_perm_parallel")
public func swiftgeo_local_perm_parallel(
    z_ptr:    UnsafePointer<Double>, z_len: Int32,
    rows_ptr: UnsafePointer<Int32>,
    cols_ptr: UnsafePointer<Int32>,
    vals_ptr: UnsafePointer<Double>, nnz: Int32, n: Int32,
    n_perm: Int32, seed: UInt64,
    out_observed: UnsafeMutablePointer<Double>,
    out_p_values: UnsafeMutablePointer<Double>,
    out_elapsed:  UnsafeMutablePointer<Double>,
    out_nthreads: UnsafeMutablePointer<Int32>
) {
    let z = Array(UnsafeBufferPointer(start: z_ptr, count: Int(z_len)))
    let r = localPermutationTest(
        z: z,
        weights: makeWeights(rows_ptr: rows_ptr, cols_ptr: cols_ptr, vals_ptr: vals_ptr, nnz: nnz, n: n),
        nPermutations: Int(n_perm), seed: seed)
    writeLocalResult(r, out_observed: out_observed, out_p_values: out_p_values,
                     out_elapsed: out_elapsed, out_nthreads: out_nthreads)
}

// MARK: - Local Geary's C

@_cdecl("swiftgeo_local_gearysc")
public func swiftgeo_local_gearysc(
    z_ptr:    UnsafePointer<Double>, z_len: Int32,
    rows_ptr: UnsafePointer<Int32>,
    cols_ptr: UnsafePointer<Int32>,
    vals_ptr: UnsafePointer<Double>, nnz: Int32, n: Int32,
    out_local: UnsafeMutablePointer<Double>
) {
    let z     = Array(UnsafeBufferPointer(start: z_ptr, count: Int(z_len)))
    let local = localGearysC(z: z, weights: makeWeights(rows_ptr: rows_ptr, cols_ptr: cols_ptr,
                                                         vals_ptr: vals_ptr, nnz: nnz, n: n))
    out_local.initialize(from: local, count: local.count)
}

@_cdecl("swiftgeo_local_perm_serial_gearysc")
public func swiftgeo_local_perm_serial_gearysc(
    z_ptr:    UnsafePointer<Double>, z_len: Int32,
    rows_ptr: UnsafePointer<Int32>,
    cols_ptr: UnsafePointer<Int32>,
    vals_ptr: UnsafePointer<Double>, nnz: Int32, n: Int32,
    n_perm: Int32, seed: UInt64,
    out_observed: UnsafeMutablePointer<Double>,
    out_p_values: UnsafeMutablePointer<Double>,
    out_elapsed:  UnsafeMutablePointer<Double>,
    out_nthreads: UnsafeMutablePointer<Int32>
) {
    let z = Array(UnsafeBufferPointer(start: z_ptr, count: Int(z_len)))
    let r = localGearysCPermutationTestSerial(
        z: z,
        weights: makeWeights(rows_ptr: rows_ptr, cols_ptr: cols_ptr, vals_ptr: vals_ptr, nnz: nnz, n: n),
        nPermutations: Int(n_perm), seed: seed)
    writeLocalResult(r, out_observed: out_observed, out_p_values: out_p_values,
                     out_elapsed: out_elapsed, out_nthreads: out_nthreads)
}

@_cdecl("swiftgeo_local_perm_parallel_gearysc")
public func swiftgeo_local_perm_parallel_gearysc(
    z_ptr:    UnsafePointer<Double>, z_len: Int32,
    rows_ptr: UnsafePointer<Int32>,
    cols_ptr: UnsafePointer<Int32>,
    vals_ptr: UnsafePointer<Double>, nnz: Int32, n: Int32,
    n_perm: Int32, seed: UInt64,
    out_observed: UnsafeMutablePointer<Double>,
    out_p_values: UnsafeMutablePointer<Double>,
    out_elapsed:  UnsafeMutablePointer<Double>,
    out_nthreads: UnsafeMutablePointer<Int32>
) {
    let z = Array(UnsafeBufferPointer(start: z_ptr, count: Int(z_len)))
    let r = localGearysCPermutationTest(
        z: z,
        weights: makeWeights(rows_ptr: rows_ptr, cols_ptr: cols_ptr, vals_ptr: vals_ptr, nnz: nnz, n: n),
        nPermutations: Int(n_perm), seed: seed)
    writeLocalResult(r, out_observed: out_observed, out_p_values: out_p_values,
                     out_elapsed: out_elapsed, out_nthreads: out_nthreads)
}

// MARK: - Spatial lag

@_cdecl("swiftgeo_spatial_lag")
public func swiftgeo_spatial_lag(
    y_ptr:    UnsafePointer<Double>, y_len: Int32,
    rows_ptr: UnsafePointer<Int32>,
    cols_ptr: UnsafePointer<Int32>,
    vals_ptr: UnsafePointer<Double>, nnz: Int32, n: Int32,
    out_wy:   UnsafeMutablePointer<Double>
) {
    let y  = Array(UnsafeBufferPointer(start: y_ptr, count: Int(y_len)))
    let wy = spatialLag(y: y, weights: makeWeights(rows_ptr: rows_ptr, cols_ptr: cols_ptr,
                                                    vals_ptr: vals_ptr, nnz: nnz, n: n))
    out_wy.initialize(from: wy, count: wy.count)
}

@_cdecl("swiftgeo_spatial_lag_parallel")
public func swiftgeo_spatial_lag_parallel(
    y_ptr:    UnsafePointer<Double>, y_len: Int32,
    rows_ptr: UnsafePointer<Int32>,
    cols_ptr: UnsafePointer<Int32>,
    vals_ptr: UnsafePointer<Double>, nnz: Int32, n: Int32,
    out_wy:   UnsafeMutablePointer<Double>
) {
    let y  = Array(UnsafeBufferPointer(start: y_ptr, count: Int(y_len)))
    let wy = spatialLagParallel(y: y, weights: makeWeights(rows_ptr: rows_ptr, cols_ptr: cols_ptr,
                                                            vals_ptr: vals_ptr, nnz: nnz, n: n))
    out_wy.initialize(from: wy, count: wy.count)
}

// MARK: - Geary's C

@_cdecl("swiftgeo_gearysc")
public func swiftgeo_gearysc(
    z_ptr:    UnsafePointer<Double>, z_len: Int32,
    rows_ptr: UnsafePointer<Int32>,
    cols_ptr: UnsafePointer<Int32>,
    vals_ptr: UnsafePointer<Double>, nnz: Int32, n: Int32
) -> Double {
    let z = Array(UnsafeBufferPointer(start: z_ptr, count: Int(z_len)))
    return gearysC(z: z, weights: makeWeights(rows_ptr: rows_ptr, cols_ptr: cols_ptr,
                                               vals_ptr: vals_ptr, nnz: nnz, n: n))
}

@_cdecl("swiftgeo_perm_serial_gearysc")
public func swiftgeo_perm_serial_gearysc(
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
    let r = permutationTestSerial(
        z: z,
        weights: makeWeights(rows_ptr: rows_ptr, cols_ptr: cols_ptr, vals_ptr: vals_ptr, nnz: nnz, n: n),
        nPermutations: Int(n_perm), seed: seed, statistic: gearysC)
    writeResult(r, out_null: out_null, out_observed: out_observed,
                out_p_value: out_p_value, out_elapsed: out_elapsed, out_nthreads: out_nthreads)
}

@_cdecl("swiftgeo_perm_parallel_gearysc")
public func swiftgeo_perm_parallel_gearysc(
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
    let r = permutationTest(
        z: z,
        weights: makeWeights(rows_ptr: rows_ptr, cols_ptr: cols_ptr, vals_ptr: vals_ptr, nnz: nnz, n: n),
        nPermutations: Int(n_perm), seed: seed, statistic: gearysC)
    writeResult(r, out_null: out_null, out_observed: out_observed,
                out_p_value: out_p_value, out_elapsed: out_elapsed, out_nthreads: out_nthreads)
}

@_cdecl("swiftgeo_perm_metal_gearysc")
public func swiftgeo_perm_metal_gearysc(
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
    guard let r = gearyCPermutationTestMetal(
        z: z,
        weights: makeWeights(rows_ptr: rows_ptr, cols_ptr: cols_ptr, vals_ptr: vals_ptr, nnz: nnz, n: n),
        nPermutations: Int(n_perm), seed: seed)
    else { return 0 }
    writeResult(r, out_null: out_null, out_observed: out_observed,
                out_p_value: out_p_value, out_elapsed: out_elapsed, out_nthreads: out_nthreads)
    return 1
}

// MARK: - Getis-Ord G (CPU only — positive z required)

@_cdecl("swiftgeo_perm_serial_getisordg")
public func swiftgeo_perm_serial_getisordg(
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
    let r = permutationTestSerial(
        z: z,
        weights: makeWeights(rows_ptr: rows_ptr, cols_ptr: cols_ptr, vals_ptr: vals_ptr, nnz: nnz, n: n),
        nPermutations: Int(n_perm), seed: seed, statistic: getisOrdG)
    writeResult(r, out_null: out_null, out_observed: out_observed,
                out_p_value: out_p_value, out_elapsed: out_elapsed, out_nthreads: out_nthreads)
}

@_cdecl("swiftgeo_perm_parallel_getisordg")
public func swiftgeo_perm_parallel_getisordg(
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
    let r = permutationTest(
        z: z,
        weights: makeWeights(rows_ptr: rows_ptr, cols_ptr: cols_ptr, vals_ptr: vals_ptr, nnz: nnz, n: n),
        nPermutations: Int(n_perm), seed: seed, statistic: getisOrdG)
    writeResult(r, out_null: out_null, out_observed: out_observed,
                out_p_value: out_p_value, out_elapsed: out_elapsed, out_nthreads: out_nthreads)
}

// MARK: - Join count (CPU only — binary z required)

@_cdecl("swiftgeo_perm_serial_joincount")
public func swiftgeo_perm_serial_joincount(
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
    let r = permutationTestSerial(
        z: z,
        weights: makeWeights(rows_ptr: rows_ptr, cols_ptr: cols_ptr, vals_ptr: vals_ptr, nnz: nnz, n: n),
        nPermutations: Int(n_perm), seed: seed, statistic: joinCount)
    writeResult(r, out_null: out_null, out_observed: out_observed,
                out_p_value: out_p_value, out_elapsed: out_elapsed, out_nthreads: out_nthreads)
}

@_cdecl("swiftgeo_perm_parallel_joincount")
public func swiftgeo_perm_parallel_joincount(
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
    let r = permutationTest(
        z: z,
        weights: makeWeights(rows_ptr: rows_ptr, cols_ptr: cols_ptr, vals_ptr: vals_ptr, nnz: nnz, n: n),
        nPermutations: Int(n_perm), seed: seed, statistic: joinCount)
    writeResult(r, out_null: out_null, out_observed: out_observed,
                out_p_value: out_p_value, out_elapsed: out_elapsed, out_nthreads: out_nthreads)
}
