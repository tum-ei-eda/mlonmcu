import logging

import pytest

from mlonmcu.environment.environment import UserEnvironment
from mlonmcu.environment.loader import load_environment_from_file


def write_yaml(tmp_path, contents):
    path = tmp_path / "environment.yml"
    path.write_text(contents, encoding="utf-8")
    return path


def test_load_environment_rejects_empty_yaml(tmp_path):
    path = write_yaml(tmp_path, "---\n")
    with pytest.raises(RuntimeError, match="Invalid YAML contents"):
        load_environment_from_file(str(path), UserEnvironment)


def test_load_environment_rejects_repository_without_url(tmp_path):
    path = write_yaml(tmp_path, "repos:\n  models:\n    ref: main\n")
    with pytest.raises(RuntimeError, match="Missing field 'url'"):
        load_environment_from_file(path, UserEnvironment)


def test_load_minimal_environment_uses_defaults(tmp_path):
    path = write_yaml(tmp_path, "home: null\n")
    env = load_environment_from_file(path, UserEnvironment)

    assert env.home is None
    assert env.defaults.log_level is None
    assert env.defaults.log_to_file is False
    assert env.defaults.log_rotate is False
    assert env.defaults.cleanup_auto is False
    assert env.defaults.cleanup_keep == 100


def test_load_complete_environment(tmp_path):
    home = tmp_path / "home"
    path = write_yaml(
        tmp_path,
        f"""
home: {home}
logging:
  level: DEBUG
  to_file: true
  rotate: true
cleanup:
  auto: true
  keep: 7
paths:
  deps: deps
  models: [models, extra-models]
repos:
  models:
    url: https://example.com/models.git
    ref: main
    options:
      recursive: false
frameworks:
  default: tvm
  tvm:
    enabled: true
    features:
      framework_feature: true
    backends:
      default: tvmaot
      tvmaot:
        enabled: true
        features:
          backend_feature: false
frontends:
  tflite:
    enabled: false
    features:
      frontend_feature: true
platforms:
  mlif:
    features:
      platform_feature: true
toolchains: [gcc, llvm]
targets:
  default: host_x86
  host_x86:
    enabled: true
    features:
      target_feature: false
vars:
  jobs: 4
flags:
  tool.path: [debug]
""",
    )

    env = load_environment_from_file(path, UserEnvironment)

    assert env.home == str(home)
    assert env.defaults.log_level == logging.DEBUG
    assert env.defaults.log_to_file is True
    assert env.defaults.log_rotate is True
    assert env.defaults.cleanup_auto is True
    assert env.defaults.cleanup_keep == 7
    assert env.defaults.default_framework == "tvm"
    assert env.defaults.default_backends == {"tvm": "tvmaot"}
    assert env.defaults.default_target == "host_x86"

    assert env.paths["deps"].path == home / "deps"
    assert [item.path for item in env.paths["models"]] == [home / "models", home / "extra-models"]
    assert env.repos["models"].url == "https://example.com/models.git"
    assert env.repos["models"].ref == "main"
    assert env.repos["models"].recursive is False

    framework = env.frameworks[0]
    assert framework.name == "tvm"
    assert framework.enabled is True
    assert framework.features[0].name == "framework_feature"
    assert framework.features[0].supported is True
    assert framework.backends[0].name == "tvmaot"
    assert framework.backends[0].features[0].name == "backend_feature"
    assert framework.backends[0].features[0].supported is False

    assert env.frontends[0].name == "tflite"
    assert env.frontends[0].enabled is False
    assert env.frontends[0].features[0].frontend == "tflite"
    assert env.platforms[0].name == "mlif"
    assert env.platforms[0].features[0].platform == "mlif"
    assert env.targets[0].name == "host_x86"
    assert env.targets[0].features[0].target == "host_x86"
    assert env.toolchains == ["gcc", "llvm"]
    assert env.vars == {"jobs": 4}
    assert env.flags == {"tool.path": ["debug"]}


def test_load_environment_component_defaults(tmp_path):
    path = write_yaml(
        tmp_path,
        """
frameworks:
  tvm:
    backends:
      tvmaot: {}
frontends:
  tflite: {}
platforms:
  mlif: {}
targets:
  host_x86: {}
""",
    )

    env = load_environment_from_file(path, UserEnvironment)

    assert env.frameworks[0].enabled is False
    assert env.frameworks[0].backends[0].enabled is True
    assert env.frontends[0].enabled is True
    assert env.platforms[0].enabled is True
    assert env.targets[0].enabled is True
