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
//   [Int32: nPerm][Float64 x nPerm: null][Float64: observed][Float64: pValue][Float64: elapsed][Int32: nThreads]

import Foundation

public enum BinaryIOError: Error {
    case fileNotFound(String)
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

        for i in 0..<nnz { rows[i]   = ptr.loadUnaligned(fromByteOffset: offset, as: Int32.self);  offset += 4 }
        for i in 0..<nnz { cols[i]   = ptr.loadUnaligned(fromByteOffset: offset, as: Int32.self);  offset += 4 }
        for i in 0..<nnz { values[i] = ptr.loadUnaligned(fromByteOffset: offset, as: Double.self); offset += 8 }

        return SparseWeights(rows: rows, cols: cols, values: values, n: n)
    }
}

// MARK: - Write results

public func writeResults(_ result: PermutationResult, to path: String) throws {
    var data = Data()

    func appendInt32(_ v: Int32)  { var x = v; data.append(contentsOf: withUnsafeBytes(of: &x) { Array($0) }) }
    func appendDouble(_ v: Double) { var x = v; data.append(contentsOf: withUnsafeBytes(of: &x) { Array($0) }) }

    appendInt32(Int32(result.nullDistribution.count))
    result.nullDistribution.forEach { appendDouble($0) }
    appendDouble(result.observed)
    appendDouble(result.pValueTwoSided)
    appendDouble(result.elapsedSeconds)
    appendInt32(Int32(result.nThreads))

    guard FileManager.default.createFile(atPath: path, contents: data) else {
        throw BinaryIOError.writeError(path)
    }
}
