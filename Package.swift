// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "s4nnc",
    platforms: [
        .macOS(.v13),
        .iOS(.v16),
        .tvOS(.v16),
        .visionOS(.v1)
    ],
    products: [
        .library(
            name: "NNC",
            targets: ["NNC"]),
        .library(
            name: "NNCCoreMLConversion",
            targets: ["NNCCoreMLConversion"]),
    ],
    dependencies: [
        // .package(url: "https://github.com/weiyanlin117/ccv.git", revision: "40f1209bca71af2cc8f63fb7f169d5e73a9bc91d"),
        .package(path: "../ccv"),
        .package(url: "https://github.com/weiyanlin117/swift-fpzip-support.git", branch: "develop"),
    ],
    targets: [
        // C_zlib - System zlib wrapper
        .target(
            name: "C_zlib",
            path: "nnc/C_zlib",
            publicHeadersPath: ".",
            cSettings: [
                .define("_GNU_SOURCE"),
            ],
            linkerSettings: [
                .linkedLibrary("z"),
            ]
        ),

        // NNC - Main Swift library
        .target(
            name: "NNC",
            dependencies: [
                .product(name: "nnc", package: "ccv"),
                .product(name: "sfmt", package: "ccv"),
                .product(name: "lib_nnc_mps_compat", package: "ccv"),
                .product(name: "C_fpzip", package: "swift-fpzip-support"),
                "C_zlib",
            ],
            path: "nnc",
            exclude: [
                "C_ccv",
                "C_nnc",
                "C_sfmt",
                "C_zlib",
                "C_sqlite3",
                "BUILD.bazel",
                "CoreMLConversion.swift",
                "PythonConversion.swift",
                "MuJoCoConversion.swift",
            ],
            sources: [
                "AnyModel.swift",
                "AutoGrad.swift",
                "DataFrame.swift",
                "DataFrameAddons.swift",
                "DataFrameCore.swift",
                "DynamicGraph.swift",
                "Functional.swift",
                "FunctionalAddons.swift",
                "GradScaler.swift",
                "Group.swift",
                "Hint.swift",
                "Loss.swift",
                "Model.swift",
                "ModelAddons.swift",
                "ModelBuilder.swift",
                "ModelCore.swift",
                "ModelIOAddons.swift",
                "Operators.swift",
                "Optimizer.swift",
                "OptimizerAddons.swift",
                "Store.swift",
                "StreamContext.swift",
                "Tensor.swift",
                "TensorGroup.swift",
                "Wrapped.swift",
            ]
        ),

        // NNCCoreMLConversion - CoreML conversion utilities
        .target(
            name: "NNCCoreMLConversion",
            dependencies: ["NNC",
                        .product(name: "lib_nnc_mps_compat", package: "ccv"),
                        .product(name: "sfmt", package: "ccv"),
            ],
            path: "nnc",
            sources: ["CoreMLConversion.swift"]
        ),

        // Test targets
        .testTarget(
            name: "NNCTests",
            dependencies: ["NNC"],
            path: "test",
            exclude: [
                "coreml",
                "python",
                "BUILD.bazel",
            ],
            sources: [
                "dataframe.swift",
                "graph.swift",
                "loss.swift",
                "model.swift",
                "ops.swift",
                "optimizer.swift",
                "store.swift",
                "tensor.swift",
            ],
            resources: [
                .copy("scaled_data.csv"),
                .copy("some_variables.db"),
            ]
        ),

        // TODO: Enable when lib_nnc_mps_compat is exposed from ccv package
        .testTarget(
            name: "NNCCoreMLTests",
            dependencies: ["NNCCoreMLConversion"],
            path: "test/coreml",
            sources: [
                "mlshapedarray.swift",
            ]
        ),
    ]
)
