// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "SwiftGeo",
    platforms: [
        .macOS(.v13)
    ],
    targets: [
        // Core library: Moran's I permutation via Accelerate + Metal
        .target(
            name: "SwiftGeo",
            path: "Sources/SwiftGeo",
            resources: [
                .process("MoranPermutation.metal")
            ],
            linkerSettings: [
                .linkedFramework("Accelerate"),
                .linkedFramework("Metal"),
                .linkedFramework("Foundation")
            ]
        ),
        // CLI runner: reads binary fixtures, runs permutations, writes results
        .executableTarget(
            name: "SwiftGeoCLI",
            dependencies: ["SwiftGeo"],
            path: "Sources/SwiftGeoCLI"
        ),
    ]
)
