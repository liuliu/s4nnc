load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")

git_repository(
	name = "ccv",
	remote = "https://github.com/liuliu/ccv.git",
	commit = "8e01e56819e1645f84837a91354f6c164875df55",
	shallow_since = "1603667789 -0400"
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
	have_libpng = True,
	have_libjpeg = True,
	have_fftw3 = True,
	have_pthread = True,
	have_liblinear = True,
	have_tesseract = True,
	have_gsl = True,
	have_cudnn = True,
	have_nccl = True,
	use_openmp = True,
	use_dispatch = True
)

git_repository(
	name = "dflat",
	remote = "https://github.com/liuliu/dflat.git",
	commit = "88aec220642bb5e416074bc8b8a4e5c8b86a61c2",
	shallow_since = "1604112303 -0400"
)

load("@dflat//:deps.bzl", "dflat_deps")

dflat_deps()

git_repository(
	name = "build_bazel_rules_swift",
	remote = "https://github.com/bazelbuild/rules_swift.git",
	commit = "6ae82f57ebefa13df5ce1daf7a2fd3080e41df55",
	shallow_since = "1599689969 -0700"
)

load("@build_bazel_rules_swift//swift:repositories.bzl", "swift_rules_dependencies")

swift_rules_dependencies()

load("@build_bazel_apple_support//lib:repositories.bzl", "apple_support_dependencies")

apple_support_dependencies()

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")

protobuf_deps()

new_git_repository(
	name = "PythonKit",
	remote = "https://github.com/pvieito/PythonKit.git",
	commit = "669eeae6e6f98b6f56c1f675f8baceeb5b2b0920",
	shallow_since = "1603358082 +0200",
	build_file = "PythonKit.BUILD"
)
