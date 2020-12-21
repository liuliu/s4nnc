workspace(name = "s4nnc")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")

git_repository(
	name = "ccv",
	remote = "https://github.com/liuliu/ccv.git",
	commit = "1b6d0e0e2329d203eb3524f562ce4459d24f36b8",
	shallow_since = "1608511703 -0500"
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
	commit = "59a868e84e1d6a5e01569cf92086554033415fa4",
	shallow_since = "1604702703 -0800",
	build_file = "PythonKit.BUILD"
)
