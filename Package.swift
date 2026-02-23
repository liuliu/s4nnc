// swift-tools-version:5.9
import PackageDescription

let package = Package(
  name: "s4nnc",
  platforms: [
    .macOS(.v13),
    .iOS(.v16),
    .tvOS(.v16),
    .visionOS(.v1),
  ],
  products: [
    .library(
      name: "NNC",
      targets: ["NNC"]),
    .library(
      name: "NNCCoreMLConversion",
      targets: ["NNCCoreMLConversion"]),
    .library(
      name: "TensorBoard",
      targets: ["TensorBoard"]),
  ],
  dependencies: [
    .package(
      url: "https://github.com/liuliu/ccv.git", revision: "2a670d8814bd8f8990f8b32879d08f4f424211ff"
    ),
    .package(
      url: "https://github.com/weiyanlin117/swift-fpzip-support.git",
      revision: "0ec6d4668c9c83bc3da0f8b2d6dfc46da0b98609"),
    .package(
      url: "https://github.com/apple/swift-protobuf.git",
      revision: "d57a5aecf24a25b32ec4a74be2f5d0a995a47c4b"),
    .package(
      url: "https://github.com/apple/swift-system.git",
      revision: "fbd61a676d79cbde05cd4fda3cc46e94d6b8f0eb"),
  ],
  targets: [
    // C_zlib - System zlib wrapper
    .systemLibrary(
      name: "C_zlib",
      path: "nnc/C_zlib",
      pkgConfig: "zlib",
      providers: [
        .brew(["zlib"]),
        .apt(["zlib1g-dev"]),
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
      dependencies: [
        "NNC",
        .product(name: "lib_nnc_mps_compat", package: "ccv"),
        .product(name: "sfmt", package: "ccv"),
      ],
      path: "nnc",
      exclude: [
        "C_ccv",
        "C_nnc",
        "C_sfmt",
        "C_zlib",
        "C_sqlite3",
        "BUILD.bazel",
        "PythonConversion.swift",
        "MuJoCoConversion.swift",
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
      ],
      sources: ["CoreMLConversion.swift"]
    ),

    // TensorBoard - TensorBoard logging support
    .target(
      name: "TensorBoard",
      dependencies: [
        "NNC",
        .product(name: "SwiftProtobuf", package: "swift-protobuf"),
        .product(name: "SystemPackage", package: "swift-system"),
      ],
      path: "tensorboard",
      exclude: ["BUILD.bazel"]
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

    .testTarget(
      name: "NNCCoreMLTests",
      dependencies: ["NNCCoreMLConversion"],
      path: "test/coreml",
      sources: [
        "mlshapedarray.swift"
      ]
    ),
  ]
)
