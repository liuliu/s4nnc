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
        commit = "ae98f0801f23c9e69be390104f86c93ebca09f93",
        shallow_since = "1651689143 -0400",
    )

    _maybe(
        new_git_repository,
        name = "PythonKit",
        remote = "https://github.com/liuliu/PythonKit.git",
        commit = "99a298f0413b0ac278ac58b7ac9045da920c347d",
        shallow_since = "1642703957 -0500",
        build_file = "@s4nnc//:external/PythonKit.BUILD",
    )
