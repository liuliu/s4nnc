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
        commit = "f4d0ee27ca87b116ff3d4f5a7264ac0d3e98e1f4",
        shallow_since = "1658510908 -0400",
    )

    _maybe(
        new_git_repository,
        name = "PythonKit",
        remote = "https://github.com/liuliu/PythonKit.git",
        commit = "99a298f0413b0ac278ac58b7ac9045da920c347d",
        shallow_since = "1642703957 -0500",
        build_file = "@s4nnc//:external/PythonKit.BUILD",
    )

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
