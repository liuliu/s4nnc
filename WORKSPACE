workspace(name = "s4nnc")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

git_repository(
    name = "ccv",
    commit = "9cd544d6eb1e9924edd48ec7f9467408cb409b07",
    remote = "https://github.com/liuliu/ccv.git",
    shallow_since = "1765613295 -0500",
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
    commit = "bffd22a56b8949616dfbd710cdca385cb2800274",
    remote = "https://github.com/bazelbuild/rules_swift.git",
    shallow_since = "1752542865 -0400",
)

load("@build_bazel_rules_swift//swift:repositories.bzl", "swift_rules_dependencies")

swift_rules_dependencies()

load("@build_bazel_rules_swift//swift:extras.bzl", "swift_rules_extra_dependencies")

swift_rules_extra_dependencies()

git_repository(
    name = "swift-mujoco",
    commit = "a37f3ba8e7a245a229c2372cf8eee492800d328a",
    remote = "https://github.com/liuliu/swift-mujoco.git",
    shallow_since = "1753395265 -0400"
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
    commit = "d57a5aecf24a25b32ec4a74be2f5d0a995a47c4b",
    remote = "https://github.com/apple/swift-protobuf.git",
    shallow_since = "1720448759 -0400",
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
    commit = "fbd61a676d79cbde05cd4fda3cc46e94d6b8f0eb",
    remote = "https://github.com/apple/swift-system.git",
    shallow_since = "1729316385 -0700",
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

http_archive(
    name = "rules_python",
    sha256 = "fa532d635f29c038a64c8062724af700c30cf6b31174dd4fac120bc561a1a560",
    strip_prefix = "rules_python-1.5.1",
    url = "https://github.com/bazel-contrib/rules_python/releases/download/1.5.1/rules_python-1.5.1.tar.gz",
)

load("@rules_python//python:repositories.bzl", "py_repositories")

py_repositories()

# buildifier is written in Go and hence needs rules_go to be built.
# See https://github.com/bazelbuild/rules_go for the up to date setup instructions.
http_archive(
    name = "io_bazel_rules_go",
    sha256 = "6dc2da7ab4cf5d7bfc7c949776b1b7c733f05e56edc4bcd9022bb249d2e2a996",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_go/releases/download/v0.39.1/rules_go-v0.39.1.zip",
        "https://github.com/bazelbuild/rules_go/releases/download/v0.39.1/rules_go-v0.39.1.zip",
    ],
)

load("@io_bazel_rules_go//go:deps.bzl", "go_rules_dependencies")

go_rules_dependencies()

load("@io_bazel_rules_go//go:deps.bzl", "go_register_toolchains")

go_register_toolchains(version = "1.20.3")

http_archive(
    name = "bazel_gazelle",
    sha256 = "727f3e4edd96ea20c29e8c2ca9e8d2af724d8c7778e7923a854b2c80952bc405",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-gazelle/releases/download/v0.30.0/bazel-gazelle-v0.30.0.tar.gz",
        "https://github.com/bazelbuild/bazel-gazelle/releases/download/v0.30.0/bazel-gazelle-v0.30.0.tar.gz",
    ],
)

load("@bazel_gazelle//:deps.bzl", "gazelle_dependencies")

# If you use WORKSPACE.bazel, use the following line instead of the bare gazelle_dependencies():
# gazelle_dependencies(go_repository_default_config = "@//:WORKSPACE.bazel")
gazelle_dependencies()

http_archive(
    name = "com_google_protobuf",
    sha256 = "3bd7828aa5af4b13b99c191e8b1e884ebfa9ad371b0ce264605d347f135d2568",
    strip_prefix = "protobuf-3.19.4",
    urls = [
        "https://github.com/protocolbuffers/protobuf/archive/v3.19.4.tar.gz",
    ],
)

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")

protobuf_deps()

http_archive(
    name = "com_github_bazelbuild_buildtools",
    sha256 = "ae34c344514e08c23e90da0e2d6cb700fcd28e80c02e23e4d5715dddcb42f7b3",
    strip_prefix = "buildtools-4.2.2",
    urls = [
        "https://github.com/bazelbuild/buildtools/archive/refs/tags/4.2.2.tar.gz",
    ],
)
