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
    commit = "4fdcff4147ddfee47ac2fed255ca23c3a829ff1f",
    shallow_since = "1605890380 -0500"
  )

  _maybe(
    new_git_repository,
    name = "PythonKit",
    remote = "https://github.com/pvieito/PythonKit.git",
    commit = "59a868e84e1d6a5e01569cf92086554033415fa4",
    shallow_since = "1604702703 -0800",
    build_file = "@s4nnc//:external/PythonKit.BUILD"
  )

