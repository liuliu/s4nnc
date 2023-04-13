package(
    default_visibility = ["//visibility:public"],
)

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
)
