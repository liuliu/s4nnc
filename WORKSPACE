load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")

git_repository(
	name = "ccv",
	remote = "https://github.com/liuliu/ccv.git",
	commit = "7e5930451b2988b4c3324d06607787b0ba848b81",
	shallow_since = "1604552787 -0500"
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
	commit = "3dc11274e8c466dd28ee35cdd04e84ddf7d420bc",
	shallow_since = "1604185591 -0400"
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
	remote = "https://github.com/liuliu/PythonKit.git",
	commit = "5d1cf3215cc232118b3e7891ba2cabb08d414e66",
	shallow_since = "1604363258 -0500",
	build_file = "PythonKit.BUILD"
)
