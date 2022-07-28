load("@build_bazel_rules_swift//swift:swift.bzl", "swift_binary", "swift_library")

cc_library(
    name = "CNIOAtomics",
    srcs = glob([
        "Sources/CNIOAtomics/src/*.c",
        "Sources/CNIOAtomics/src/*.h",
    ]),
    hdrs = glob(["Sources/CNIOAtomics/include/*.h"]),
    includes = [
        "Sources/CNIOAtomics/include/",
    ],
    tags = ["swift_module=CNIOAtomics"],
)

cc_library(
    name = "CNIOSHA1",
    srcs = glob([
        "Sources/CNIOSHA1/*.c",
        "Sources/CNIOSHA1/*.h",
    ]),
    hdrs = glob(["Sources/CNIOSHA1/include/*.h"]),
    includes = [
        "Sources/CNIOSHA1/include/",
    ],
    tags = ["swift_module=CNIOSHA1"],
)

cc_library(
    name = "CNIOLinux",
    srcs = glob([
        "Sources/CNIOLinux/*.c",
        "Sources/CNIOLinux/*.h",
    ]),
    hdrs = glob(["Sources/CNIOLinux/include/*.h"]),
    includes = [
        "Sources/CNIOLinux/include/",
    ],
    tags = ["swift_module=CNIOLinux"],
)

cc_library(
    name = "CNIODarwin",
    srcs = glob([
        "Sources/CNIODarwin/*.c",
        "Sources/CNIODarwin/*.h",
    ]),
    hdrs = glob(["Sources/CNIODarwin/include/*.h"]),
    defines = ["__APPLE_USE_RFC_3542"],
    includes = [
        "Sources/CNIODarwin/include/",
    ],
    tags = ["swift_module=CNIODarwin"],
)

cc_library(
    name = "CNIOWindows",
    srcs = glob([
        "Sources/CNIOWindows/*.c",
        "Sources/CNIOWindows/*.h",
    ]),
    hdrs = glob(["Sources/CNIOWindows/include/*.h"]),
    includes = [
        "Sources/CNIOWindows/include/",
    ],
    tags = ["swift_module=CNIOWindows"],
)

swift_library(
    name = "NIOConcurrencyHelpers",
    srcs = glob([
        "Sources/NIOConcurrencyHelpers/**/*.swift",
    ]),
    module_name = "NIOConcurrencyHelpers",
    visibility = ["//visibility:public"],
    deps = [":CNIOAtomics"],
)

swift_library(
    name = "NIOCore",
    srcs = glob([
        "Sources/NIOCore/**/*.swift",
    ]),
    module_name = "NIOCore",
    visibility = ["//visibility:public"],
    deps = [
        ":CNIOLinux",
        ":CNIOWindows",
        ":NIOConcurrencyHelpers",
    ],
)

swift_library(
    name = "_NIODataStructures",
    srcs = glob([
        "Sources/_NIODataStructures/**/*.swift",
    ]),
    module_name = "_NIODataStructures",
)

swift_library(
    name = "NIOEmbedded",
    srcs = glob([
        "Sources/NIOEmbedded/**/*.swift",
    ]),
    module_name = "NIOEmbedded",
    visibility = ["//visibility:public"],
    deps = [
        ":NIOConcurrencyHelpers",
        ":NIOCore",
        ":_NIODataStructures",
        "@swift-atomics//:SwiftAtomics",
    ],
)

swift_library(
    name = "NIOPosix",
    srcs = glob([
        "Sources/NIOPosix/**/*.swift",
    ]),
    module_name = "NIOPosix",
    visibility = ["//visibility:public"],
    deps = [
        ":CNIODarwin",
        ":CNIOLinux",
        ":CNIOWindows",
        ":NIOConcurrencyHelpers",
        ":NIOCore",
        ":_NIODataStructures",
        "@swift-atomics//:SwiftAtomics",
    ],
)

swift_library(
    name = "NIO",
    srcs = glob([
        "Sources/NIO/**/*.swift",
    ]),
    module_name = "NIO",
    visibility = ["//visibility:public"],
    deps = [
        ":NIOCore",
        ":NIOEmbedded",
        ":NIOPosix",
    ],
)

swift_library(
    name = "NIOFoundationCompat",
    srcs = glob([
        "Sources/NIOFoundationCompat/**/*.swift",
    ]),
    module_name = "NIOFoundationCompat",
    visibility = ["//visibility:public"],
    deps = [
        ":NIO",
        ":NIOCore",
    ],
)

cc_library(
    name = "CNIOHTTPParser",
    srcs = glob([
        "Sources/CNIOHTTPParser/*.c",
        "Sources/CNIOHTTPParser/*.h",
    ]),
    hdrs = glob(["Sources/CNIOHTTPParser/include/*.h"]),
    includes = [
        "Sources/CNIOWindows/include/",
    ],
    tags = ["swift_module=CNIOHTTPParser"],
)

swift_library(
    name = "NIOHTTP1",
    srcs = glob([
        "Sources/NIOHTTP1/**/*.swift",
    ]),
    module_name = "NIOHTTP1",
    visibility = ["//visibility:public"],
    deps = [
        ":CNIOHTTPParser",
        ":NIO",
        ":NIOConcurrencyHelpers",
        ":NIOCore",
    ],
)

swift_library(
    name = "NIOTLS",
    srcs = glob([
        "Sources/NIOTLS/**/*.swift",
    ]),
    module_name = "NIOTLS",
    visibility = ["//visibility:public"],
    deps = [
        ":NIO",
        ":NIOCore",
    ],
)

swift_library(
    name = "NIOWebSocket",
    srcs = glob([
        "Sources/NIOWebSocket/**/*.swift",
    ]),
    module_name = "NIOWebSocket",
    visibility = ["//visibility:public"],
    deps = [
        ":CNIOSHA1",
        ":NIO",
        ":NIOCore",
        ":NIOHTTP1",
    ],
)

swift_binary(
    name = "NIOHTTP1Server",
    srcs = glob([
        "Sources/NIOHTTP1Server/**/*.swift",
    ]),
    deps = [
        ":NIOConcurrencyHelpers",
        ":NIOCore",
        ":NIOHTTP1",
        ":NIOPosix",
    ],
)
