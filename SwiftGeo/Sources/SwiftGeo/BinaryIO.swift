// BinaryIO.swift
// Read/write binary fixtures shared between Python and Swift.
//
// Format for z vector:
//   [Int32: n][Float64 x n: values]
//
// Format for weights (COO):
//   [Int32: n][Int32: nnz][Int32 x nnz: rows][Int32 x nnz: cols][Float64 x nnz: values]
//
// Format for null distribution output:
//   [Int32: nPerm][Float64 x nPerm: values][Float64: observed][Float64: pValue][Float64: elapsed]

import Foundation

public enum BinaryIOError: Error {
    case fileNotFound(String)
    case readError(String)
    case writeError(String)
}

// MARK: - Read z vector

public func readZVector(from path: String) throws -> [Double] {
    guard let data = FileManager.default.contents(atPath: path) else {
        throw BinaryIOError.fileNotFound(path)
    }
    return data.withUnsafeBytes { ptr -> [Double] in
        var offset = 0
        let n = Int(ptr.loadUnaligned(fromByteOffset: offset, as: Int32.self))
        offset += 4
        var result = [Double](repeating: 0, count: n)
        for i in 0..<n {
            result[i] = ptr.loadUnaligned(fromByteOffset: offset, as: Double.self)
            offset += 8
        }
        return result
    }
}

// MARK: - Read sparse weights

public func readSparseWeights(from path: String) throws -> SparseWeights {
    guard let data = FileManager.default.contents(atPath: path) else {
        throw BinaryIOError.fileNotFound(path)
    }
    return data.withUnsafeBytes { ptr -> SparseWeights in
        var offset = 0
        let n   = Int(ptr.loadUnaligned(fromByteOffset: offset, as: Int32.self)); offset += 4
        let nnz = Int(ptr.loadUnaligned(fromByteOffset: offset, as: Int32.self)); offset += 4

        var rows   = [Int32](repeating: 0, count: nnz)
        var cols   = [Int32](repeating: 0, count: nnz)
        var values = [Double](repeating: 0, count: nnz)

        for i in 0..<nnz {
            rows[i] = ptr.loadUnaligned(fromByteOffset: offset, as: Int32.self); offset += 4
        }
        for i in 0..<nnz {
            cols[i] = ptr.loadUnaligned(fromByteOffset: offset, as: Int32.self); offset += 4
        }
        for i in 0..<nnz {
            values[i] = ptr.loadUnaligned(fromByteOffset: offset, as: Double.self); offset += 8
        }

        return SparseWeights(rows: rows, cols: cols, values: values, n: n)
    }
}

// MARK: - Write results

public func writeResults(_ result: PermutationResult, to path: String) throws {
    var data = Data()

    // nPerm
    var nPerm = Int32(result.nullDistribution.count)
    data.append(contentsOf: withUnsafeBytes(of: &nPerm) { Array($0) })

    // null distribution
    for var v in result.nullDistribution {
        data.append(contentsOf: withUnsafeBytes(of: &v) { Array($0) })
    }

    // observed, pValue, elapsed
    var obs  = result.observed
    var pval = result.pValueTwoSided
    var elapsed = result.elapsedSeconds
    data.append(contentsOf: withUnsafeBytes(of: &obs)     { Array($0) })
    data.append(contentsOf: withUnsafeBytes(of: &pval)    { Array($0) })
    data.append(contentsOf: withUnsafeBytes(of: &elapsed) { Array($0) })

    guard FileManager.default.createFile(atPath: path, contents: data) else {
        throw BinaryIOError.writeError(path)
    }
}
