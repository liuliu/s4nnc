load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library", "swift_interop_hint")

cc_library(
    name = "_NumericsShims",
    srcs = ["Sources/_NumericsShims/_NumericsShims.c"],
    hdrs = ["Sources/_NumericsShims/include/_NumericsShims.h"],
    includes = [
        "Sources/_NumericsShims/include/",
    ],
    tags = ["swift_module=_NumericsShims"],
    aspect_hints = [":NumericsShims_swift_interop"],
)

swift_interop_hint(
    name = "NumericsShims_swift_interop",
    module_name = "NumericsShims",
)

swift_library(
    name = "RealModule",
    srcs = glob([
        "Sources/RealModule/**/*.swift",
    ]),
    module_name = "RealModule",
    visibility = ["//visibility:public"],
    deps = [
        ":_NumericsShims",
    ],
)

swift_library(
    name = "ComplexModule",
    srcs = glob([
        "Sources/ComplexModule/**/*.swift",
    ]),
    module_name = "ComplexModule",
    visibility = ["//visibility:public"],
    deps = [
        ":RealModule",
    ],
)

swift_library(
    name = "Numerics",
    srcs = glob([
        "Sources/Numerics/**/*.swift",
    ]),
    module_name = "Numerics",
    visibility = ["//visibility:public"],
    deps = [
        ":ComplexModule",
        ":RealModule",
    ],
)
