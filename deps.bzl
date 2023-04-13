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
        commit = "077204465d27cbf063c9fda0ea3fb82cdd565f60",
        shallow_since = "1681410223 -0400",
    )

    _maybe(
        new_git_repository,
        name = "PythonKit",
        remote = "https://github.com/liuliu/PythonKit.git",
        commit = "fbf22756c91d89b0f2e39a89b690aaa538cf9b03",
        shallow_since = "1664547636 -0400",
        build_file = "@s4nnc//:external/PythonKit.BUILD",
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
        commit = "836bc4557b74fe6d2660218d56e3ce96aff76574",
        remote = "https://github.com/apple/swift-system.git",
        shallow_since = "1638472952 -0800",
    )

    _maybe(
        new_git_repository,
        name = "SwiftProtobuf",
        build_file = "@s4nnc//:external/swift-protobuf.BUILD",
        commit = "7cbb5279dd7e997c8f0f5537e46d4513be894ff1",
        remote = "https://github.com/apple/swift-protobuf.git",
        shallow_since = "1658527939 -0700",
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
