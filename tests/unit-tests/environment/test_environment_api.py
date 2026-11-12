import pytest

from mlonmcu.environment.config import (
    BackendConfig,
    BackendFeatureConfig,
    DefaultsConfig,
    FrameworkConfig,
    FrameworkFeatureConfig,
    FrontendConfig,
    FrontendFeatureConfig,
    PlatformConfig,
    PlatformFeatureConfig,
    TargetConfig,
    TargetFeatureConfig,
)
from mlonmcu.environment.environment import UserEnvironment
from mlonmcu.feature.type import FeatureType


@pytest.fixture()
def environment(tmp_path):
    tvmaot = BackendConfig(
        "tvmaot",
        features=[BackendFeatureConfig("debug", backend="tvmaot"), BackendFeatureConfig("shared", "tvmaot")],
    )
    disabled_backend = BackendConfig("disabled_backend", enabled=False)
    tvm = FrameworkConfig(
        "tvm",
        backends=[tvmaot, disabled_backend],
        features=[FrameworkFeatureConfig("shared", framework="tvm")],
    )
    disabled_framework = FrameworkConfig("disabled_framework", enabled=False, backends=[BackendConfig("other")])
    return UserEnvironment(
        str(tmp_path),
        defaults=DefaultsConfig(
            default_framework="tvm",
            default_backends={"tvm": "tvmaot"},
            default_target="host_x86",
        ),
        paths={"models": tmp_path / "models"},
        frameworks=[tvm, disabled_framework],
        frontends=[
            FrontendConfig("tflite", features=[FrontendFeatureConfig("shared", frontend="tflite")]),
            FrontendConfig("disabled_frontend", enabled=False),
        ],
        platforms=[
            PlatformConfig("mlif", features=[PlatformFeatureConfig("shared", platform="mlif")]),
            PlatformConfig("disabled_platform", enabled=False),
        ],
        toolchains={"gcc": True, "llvm": False},
        targets=[
            TargetConfig("host_x86", features=[TargetFeatureConfig("shared", target="host_x86", supported=False)]),
            TargetConfig("disabled_target", enabled=False),
        ],
        variables={"jobs": 4},
    )


def test_environment_basic_lookups(environment):
    assert environment.home is not None
    assert environment.lookup_path("models").name == "models"
    assert environment.lookup_var("jobs") == 4
    assert environment.lookup_var("missing", 2) == 2
    assert "UserEnvironment" in str(environment)
    with pytest.raises(AssertionError, match="Unable to find"):
        environment.lookup_path("missing")


@pytest.mark.parametrize(
    "lookup,expected",
    [
        ("lookup_framework_configs", ["tvm"]),
        ("lookup_frontend_configs", ["tflite"]),
        ("lookup_platform_configs", ["mlif"]),
        ("lookup_target_configs", ["host_x86"]),
    ],
)
def test_component_lookups_filter_disabled_entries(environment, lookup, expected):
    method = getattr(environment, lookup)
    assert method(names_only=True) == expected
    assert method(expected[0], names_only=True) == expected
    assert method("missing", names_only=True) == []


def test_backend_lookups_filter_by_framework_and_backend(environment):
    assert environment.lookup_backend_configs(names_only=True) == ["tvmaot"]
    assert environment.lookup_backend_configs(framework="tvm", names_only=True) == ["tvmaot"]
    assert environment.lookup_backend_configs(backend="tvmaot", names_only=True) == ["tvmaot"]
    assert environment.lookup_backend_configs(backend="missing") == []
    assert environment.has_backend("none") is True


def test_component_presence_checks(environment):
    assert environment.has_framework("tvm") is True
    assert environment.has_frontend("disabled_frontend") is False
    assert environment.has_platform("mlif") is True
    assert environment.has_target("missing") is False
    assert environment.has_toolchain("gcc") is True
    assert environment.has_toolchain("llvm") is False


def test_feature_lookups_by_kind_and_component(environment):
    assert [item.name for item in environment.lookup_frontend_feature_configs(frontend="tflite")] == ["shared"]
    assert [item.name for item in environment.lookup_framework_feature_configs(framework="tvm")] == ["shared"]
    assert [item.name for item in environment.lookup_backend_feature_configs(framework="tvm", backend="tvmaot")] == [
        "debug",
        "shared",
    ]
    assert [item.name for item in environment.lookup_backend_feature_configs(backend="tvmaot", name="debug")] == [
        "debug"
    ]
    assert [item.name for item in environment.lookup_platform_feature_configs(platform="mlif")] == ["shared"]
    assert [item.name for item in environment.lookup_target_feature_configs(target="host_x86")] == ["shared"]
    assert len(environment.lookup_feature_configs(name="shared")) == 5
    assert len(environment.lookup_feature_configs(name="shared", kind=FeatureType.BACKEND)) == 1
    assert environment.supports_feature("shared") is True
    assert environment.has_feature("missing") is False


def test_default_component_selection(environment):
    assert environment.get_default_frameworks() == ["tvm"]
    assert environment.get_default_backends("tvm") == ["tvmaot"]
    assert environment.get_default_backends("missing") == []
    assert environment.get_default_targets() == ["host_x86"]

    environment.defaults.default_framework = "*"
    environment.defaults.default_backends["tvm"] = "*"
    environment.defaults.default_target = "*"
    assert environment.get_default_frameworks() == ["tvm"]
    assert environment.get_default_backends("tvm") == ["tvmaot"]
    assert environment.get_default_targets() == ["host_x86"]
