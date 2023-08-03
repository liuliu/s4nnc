workspace(name = "s4nnc")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

git_repository(
    name = "ccv",
    commit = "1f2f94b686ad02d3a4766123f86270bb5445fb82",
    remote = "https://github.com/liuliu/ccv.git",
    shallow_since = "1691103606 -0400",
)

load("@ccv//config:ccv.bzl", "ccv_deps", "ccv_setting")

ccv_deps()

load("@build_bazel_rules_cuda//gpus:cuda_configure.bzl", "cuda_configure")
load("@build_bazel_rules_cuda//nccl:nccl_configure.bzl", "nccl_configure")

cuda_configure(name = "local_config_cuda")

nccl_configure(name = "local_config_nccl")

ccv_setting(
    name = "local_config_ccv",
    have_cblas = True,
    have_cudnn = True,
    have_fftw3 = True,
    have_gsl = True,
    have_libjpeg = True,
    have_libpng = True,
    have_nccl = True,
    have_pthread = True,
    use_dispatch = True,
    use_openmp = True,
)

git_repository(
    name = "build_bazel_rules_swift",
    commit = "3bc7bc164020a842ae08e0cf071ed35f0939dd39",
    remote = "https://github.com/bazelbuild/rules_swift.git",
    shallow_since = "1654173801 -0500",
)

load("@build_bazel_rules_swift//swift:repositories.bzl", "swift_rules_dependencies")

swift_rules_dependencies()

load("@build_bazel_rules_swift//swift:extras.bzl", "swift_rules_extra_dependencies")

swift_rules_extra_dependencies()

git_repository(
    name = "swift-mujoco",
    commit = "3c0a4496bd3b984fb5ddceebc8f8def7c698bd2f",
    remote = "https://github.com/liuliu/swift-mujoco.git",
    shallow_since = "1658628786 -0400"
)

load("@swift-mujoco//:deps.bzl", "swift_mujoco_deps")

swift_mujoco_deps()

git_repository(
    name = "swift-jupyter",
    commit = "22bdd9758c9070a1de38c8538b34b4cc9ec559c0",
    remote = "https://github.com/liuliu/swift-jupyter.git",
    shallow_since = "1659044971 -0400",
)

new_git_repository(
    name = "PythonKit",
    build_file = "PythonKit.BUILD",
    commit = "fbf22756c91d89b0f2e39a89b690aaa538cf9b03",
    remote = "https://github.com/liuliu/PythonKit.git",
    shallow_since = "1664547636 -0400",
)

new_git_repository(
    name = "fpzip",
    build_file = "fpzip.BUILD",
    commit = "79aa1b1bd5a0b9497b8ad4352d8561ab17113cdf",
    remote = "https://github.com/LLNL/fpzip.git",
    shallow_since = "1591380432 -0700",
)

new_git_repository(
    name = "swift-atomics",
    build_file = "swift-atomics.BUILD",
    commit = "088df27f0683f2b458021ebf04873174b91ae597",
    remote = "https://github.com/apple/swift-atomics.git",
    shallow_since = "1649274362 -0700",
)

new_git_repository(
    name = "SwiftNIO",
    build_file = "swift-nio.BUILD",
    commit = "48916a49afedec69275b70893c773261fdd2cfde",
    remote = "https://github.com/apple/swift-nio.git",
    shallow_since = "1657195654 +0100",
)

new_git_repository(
    name = "SwiftProtobuf",
    build_file = "swift-protobuf.BUILD",
    commit = "7cbb5279dd7e997c8f0f5537e46d4513be894ff1",
    remote = "https://github.com/apple/swift-protobuf.git",
    shallow_since = "1658527939 -0700",
)

new_git_repository(
    name = "SwiftArgumentParser",
    build_file = "swift-argument-parser.BUILD",
    commit = "9f39744e025c7d377987f30b03770805dcb0bcd1",
    remote = "https://github.com/apple/swift-argument-parser.git",
    shallow_since = "1661571047 -0500",
)

new_git_repository(
    name = "SwiftSystem",
    build_file = "swift-system.BUILD",
    commit = "025bcb1165deab2e20d4eaba79967ce73013f496",
    remote = "https://github.com/apple/swift-system.git",
    shallow_since = "1654977448 -0700",
)

new_git_repository(
    name = "SwiftToolsSupportCore",
    build_file = "swift-tools-support-core.BUILD",
    commit = "286b48b1d73388e1d49b2bb33aabf995838104e3",
    remote = "https://github.com/apple/swift-tools-support-core.git",
    shallow_since = "1670947584 -0800",
)

new_git_repository(
    name = "SwiftSyntax",
    build_file = "swift-syntax.BUILD",
    commit = "cd793adf5680e138bf2bcbaacc292490175d0dcd",
    remote = "https://github.com/apple/swift-syntax.git",
    shallow_since = "1676877517 +0100",
)

new_git_repository(
    name = "SwiftFormat",
    build_file = "swift-format.BUILD",
    commit = "9f1cc7172f100118229644619ce9c8f9ebc1032c",
    remote = "https://github.com/apple/swift-format.git",
    shallow_since = "1676404655 +0000",
)

new_git_repository(
    name = "SwiftNumerics",
    build_file = "swift-numerics.BUILD",
    commit = "4a2cbc186b1f8cbbc1ace12cef43d65784b2559e",
    remote = "https://github.com/apple/swift-numerics.git",
    shallow_since = "1605460976 -0500",
)

new_git_repository(
    name = "SwiftAlgorithms",
    build_file = "swift-algorithms.BUILD",
    commit = "195e0316d7ba71e134d0f6c677f64b4db6160c46",
    remote = "https://github.com/apple/swift-algorithms.git",
    shallow_since = "1645643239 -0600",
)

# buildifier is written in Go and hence needs rules_go to be built.
# See https://github.com/bazelbuild/rules_go for the up to date setup instructions.

http_archive(
    name = "io_bazel_rules_go",
    sha256 = "099a9fb96a376ccbbb7d291ed4ecbdfd42f6bc822ab77ae6f1b5cb9e914e94fa",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_go/releases/download/v0.35.0/rules_go-v0.35.0.zip",
        "https://github.com/bazelbuild/rules_go/releases/download/v0.35.0/rules_go-v0.35.0.zip",
    ],
)

load("@io_bazel_rules_go//go:deps.bzl", "go_register_toolchains", "go_rules_dependencies")

go_rules_dependencies()

go_register_toolchains(version = "1.19.1")

http_archive(
    name = "bazel_gazelle",
    sha256 = "501deb3d5695ab658e82f6f6f549ba681ea3ca2a5fb7911154b5aa45596183fa",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-gazelle/releases/download/v0.26.0/bazel-gazelle-v0.26.0.tar.gz",
        "https://github.com/bazelbuild/bazel-gazelle/releases/download/v0.26.0/bazel-gazelle-v0.26.0.tar.gz",
    ],
)

load("@bazel_gazelle//:deps.bzl", "gazelle_dependencies")

gazelle_dependencies()

git_repository(
    name = "com_github_bazelbuild_buildtools",
    commit = "174cbb4ba7d15a3ad029c2e4ee4f30ea4d76edce",
    remote = "https://github.com/bazelbuild/buildtools.git",
    shallow_since = "1607975103 +0100",
)
