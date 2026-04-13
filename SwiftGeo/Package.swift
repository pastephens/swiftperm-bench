// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "SwiftGeo",
    platforms: [
        .macOS(.v13)
    ],
    targets: [
        // Core library: Moran's I permutation via Accelerate
        .target(
            name: "SwiftGeo",
            path: "Sources/SwiftGeo",
            linkerSettings: [
                .linkedFramework("Accelerate")
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
