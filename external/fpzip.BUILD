load("@build_bazel_rules_swift//swift:swift.bzl", "swift_interop_hint")

cc_library(
    name = "fpzip",
    srcs = glob([
        "src/**/*.cpp",
        "src/**/*.h",
        "src/**/*.inl",
    ]),
    hdrs = ["include/fpzip.h"],
    defines = ["FPZIP_FP=FPZIP_FP_FAST"],
    includes = ["include"],
    local_defines = ["FPZIP_BLOCK_SIZE=0x1000"],
    tags = ["swift_module=C_fpzip"],
    visibility = ["//visibility:public"],
    aspect_hints = [":C_fpzip_swift_interop"],
)

swift_interop_hint(
    name = "C_fpzip_swift_interop",
    module_name = "C_fpzip",
)
