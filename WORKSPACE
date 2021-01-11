workspace(name = "s4nnc")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

git_repository(
    name = "ccv",
    commit = "0f10d2dd6fe2ec93433587c4b6fed52e1697ede1",
    remote = "https://github.com/liuliu/ccv.git",
    shallow_since = "1610400173 -0500",
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
    have_liblinear = True,
    have_libpng = True,
    have_nccl = True,
    have_pthread = True,
    have_tesseract = True,
    use_dispatch = True,
    use_openmp = True,
)

git_repository(
    name = "build_bazel_rules_swift",
    commit = "6ae82f57ebefa13df5ce1daf7a2fd3080e41df55",
    remote = "https://github.com/bazelbuild/rules_swift.git",
    shallow_since = "1599689969 -0700",
)

load("@build_bazel_rules_swift//swift:repositories.bzl", "swift_rules_dependencies")

swift_rules_dependencies()

load("@build_bazel_apple_support//lib:repositories.bzl", "apple_support_dependencies")

apple_support_dependencies()

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")

protobuf_deps()

new_git_repository(
    name = "PythonKit",
    build_file = "PythonKit.BUILD",
    commit = "59a868e84e1d6a5e01569cf92086554033415fa4",
    remote = "https://github.com/pvieito/PythonKit.git",
    shallow_since = "1604702703 -0800",
)

new_git_repository(
    name = "SwiftArgumentParser",
    build_file = "swift-argument-parser.BUILD",
    commit = "4273ad222e6c51969e8585541f9da5187ad94e47",
    remote = "https://github.com/apple/swift-argument-parser.git",
    shallow_since = "1607637753 -0600",
)

new_git_repository(
    name = "SwiftSyntax",
    build_file = "swift-syntax.BUILD",
    commit = "844574d683f53d0737a9c6d706c3ef31ed2955eb",
    remote = "https://github.com/apple/swift-syntax.git",
    shallow_since = "1600388447 -0700",
)

new_git_repository(
    name = "SwiftFormat",
    build_file = "swift-format.BUILD",
    commit = "12089179aa1668a2478b2b2111d98fa37f3531e3",
    remote = "https://github.com/apple/swift-format.git",
    shallow_since = "1600489295 -0700",
)

# buildifier is written in Go and hence needs rules_go to be built.
# See https://github.com/bazelbuild/rules_go for the up to date setup instructions.

http_archive(
    name = "io_bazel_rules_go",
    sha256 = "d1ffd055969c8f8d431e2d439813e42326961d0942bdf734d2c95dc30c369566",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_go/releases/download/v0.24.5/rules_go-v0.24.5.tar.gz",
        "https://github.com/bazelbuild/rules_go/releases/download/v0.24.5/rules_go-v0.24.5.tar.gz",
    ],
)

load("@io_bazel_rules_go//go:deps.bzl", "go_register_toolchains", "go_rules_dependencies")

go_rules_dependencies()

go_register_toolchains()

http_archive(
    name = "bazel_gazelle",
    sha256 = "b85f48fa105c4403326e9525ad2b2cc437babaa6e15a3fc0b1dbab0ab064bc7c",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-gazelle/releases/download/v0.22.2/bazel-gazelle-v0.22.2.tar.gz",
        "https://github.com/bazelbuild/bazel-gazelle/releases/download/v0.22.2/bazel-gazelle-v0.22.2.tar.gz",
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
