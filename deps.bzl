load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")

def _maybe(repo_rule, name, **kwargs):
    """Executes the given repository rule if it hasn't been executed already.
    Args:
      repo_rule: The repository rule to be executed (e.g., `http_archive`.)
      name: The name of the repository to be defined by the rule.
      **kwargs: Additional arguments passed directly to the repository rule.
    """
    if not native.existing_rule(name):
        repo_rule(name = name, **kwargs)

def s4nnc_deps():
    """Loads common dependencies needed to compile the s4nnc library."""

    _maybe(
        git_repository,
        name = "ccv",
        remote = "https://github.com/liuliu/ccv.git",
        commit = "73206d2f1f7458e9c1482e5f84b28c99de13c079",
        shallow_since = "1753384601 -0400",
    )

    _maybe(
        new_git_repository,
        name = "PythonKit",
        remote = "https://github.com/liuliu/PythonKit.git",
        commit = "fbf22756c91d89b0f2e39a89b690aaa538cf9b03",
        shallow_since = "1664547636 -0400",
        build_file = "@s4nnc//:external/PythonKit.BUILD",
    )

    _maybe(
        new_git_repository,
        name = "fpzip",
        commit = "79aa1b1bd5a0b9497b8ad4352d8561ab17113cdf",
        remote = "https://github.com/LLNL/fpzip.git",
        shallow_since = "1591380432 -0700",
        build_file = "@s4nnc//:external/fpzip.BUILD",
    )

def s4nnc_extra_deps():
    """Loads common dependencies needed to compile gym and tensorboard."""

    _maybe(
        new_git_repository,
        name = "SwiftNumerics",
        remote = "https://github.com/apple/swift-numerics.git",
        commit = "4a2cbc186b1f8cbbc1ace12cef43d65784b2559e",
        shallow_since = "1605460976 -0500",
        build_file = "@s4nnc//:external/swift-numerics.BUILD",
    )

    _maybe(
        new_git_repository,
        name = "SwiftAlgorithms",
        remote = "https://github.com/apple/swift-algorithms.git",
        commit = "195e0316d7ba71e134d0f6c677f64b4db6160c46",
        shallow_since = "1645643239 -0600",
        build_file = "@s4nnc//:external/swift-algorithms.BUILD",
    )

    _maybe(
        new_git_repository,
        name = "SwiftSystem",
        build_file = "@s4nnc//:external/swift-system.BUILD",
        commit = "fbd61a676d79cbde05cd4fda3cc46e94d6b8f0eb",
        remote = "https://github.com/apple/swift-system.git",
        shallow_since = "1729316385 -0700",
    )

    _maybe(
        new_git_repository,
        name = "SwiftProtobuf",
        build_file = "@s4nnc//:external/swift-protobuf.BUILD",
        commit = "d57a5aecf24a25b32ec4a74be2f5d0a995a47c4b",
        remote = "https://github.com/apple/swift-protobuf.git",
        shallow_since = "1720448759 -0400",
    )

    _maybe(
        git_repository,
        name = "swift-jupyter",
        commit = "22bdd9758c9070a1de38c8538b34b4cc9ec559c0",
        remote = "https://github.com/liuliu/swift-jupyter.git",
        shallow_since = "1659044971 -0400",
    )

    _maybe(
        new_git_repository,
        name = "swift-atomics",
        build_file = "@s4nnc//:external/swift-atomics.BUILD",
        commit = "088df27f0683f2b458021ebf04873174b91ae597",
        remote = "https://github.com/apple/swift-atomics.git",
        shallow_since = "1649274362 -0700",
    )

    _maybe(
        new_git_repository,
        name = "SwiftNIO",
        build_file = "@s4nnc//:external/swift-nio.BUILD",
        commit = "48916a49afedec69275b70893c773261fdd2cfde",
        remote = "https://github.com/apple/swift-nio.git",
        shallow_since = "1657195654 +0100",
    )
