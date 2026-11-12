import pytest

from mlonmcu.feature.feature import (
    BackendFeature,
    Feature,
    FrameworkFeature,
    FrontendFeature,
    PlatformFeature,
    RunFeature,
    SetupFeature,
    TargetFeature,
)
from mlonmcu.feature.type import FeatureType


def test_feature_base_configuration_and_representation():
    feature = Feature("demo", config={"demo.enabled": "false", "demo.option": 1})
    assert feature.enabled is False
    assert repr(feature) == "Feature(demo)"
    assert feature.remove_config_prefix({"demo.a": 1, "other.a": 2}) == {"a": 1}
    assert Feature.types() == [None]
    assert ConfigTargetFeature.types() == [FeatureType.TARGET]


class ConfigFrontendFeature(FrontendFeature):
    def get_frontend_config(self, frontend):
        return {f"{frontend}.enabled": True}


class ConfigFrameworkFeature(FrameworkFeature):
    def get_framework_config(self, framework):
        return {f"{framework}.enabled": True}


class ConfigBackendFeature(BackendFeature):
    def get_backend_config(self, backend):
        return {f"{backend}.enabled": True}


class ConfigTargetFeature(TargetFeature):
    def get_target_config(self, target):
        return {f"{target}.enabled": True}

    def get_target_callbacks(self, target):
        return (lambda: f"pre-{target}"), (lambda: f"post-{target}")


class ConfigPlatformFeature(PlatformFeature):
    def get_platform_config(self, platform):
        return {f"{platform}.enabled": True}

    def get_platform_defs(self, platform):
        return {"PLATFORM": platform}


class ConfigRunFeature(RunFeature):
    def get_run_config(self):
        return {"run.enabled": True}


@pytest.mark.parametrize(
    "feature,method,argument,expected",
    [
        (ConfigFrontendFeature("frontend"), "add_frontend_config", "tflite", {"tflite.enabled": True}),
        (ConfigFrameworkFeature("framework"), "add_framework_config", "tvm", {"tvm.enabled": True}),
        (ConfigBackendFeature("backend"), "add_backend_config", "tvmaot", {"tvmaot.enabled": True}),
        (ConfigTargetFeature("target"), "add_target_config", "host", {"host.enabled": True}),
        (ConfigPlatformFeature("platform"), "add_platform_config", "mlif", {"mlif.enabled": True}),
    ],
)
def test_component_features_extend_configuration(feature, method, argument, expected):
    config = {"existing": 1}
    getattr(feature, method)(argument, config)
    assert config == {"existing": 1, **expected}


def test_base_component_features_return_empty_configuration():
    assert FrontendFeature("frontend").get_frontend_config("tflite") == {}
    assert FrameworkFeature("framework").get_framework_config("tvm") == {}
    assert BackendFeature("backend").get_backend_config("tvmaot") == {}
    assert TargetFeature("target").get_target_config("host") == {}
    assert PlatformFeature("platform").get_platform_config("mlif") == {}
    assert PlatformFeature("platform").get_platform_defs("mlif") == {}
    assert RunFeature("run").get_run_config() == {}


def test_target_feature_adds_available_callbacks():
    feature = ConfigTargetFeature("target")
    pre_callbacks, post_callbacks = [], []
    feature.add_target_callbacks("host", pre_callbacks, post_callbacks)
    assert pre_callbacks[0]() == "pre-host"
    assert post_callbacks[0]() == "post-host"

    TargetFeature("target").add_target_callbacks("host", pre_callbacks, post_callbacks)
    feature.add_target_callbacks("host", None, None)
    assert len(pre_callbacks) == len(post_callbacks) == 1


def test_platform_and_run_features_extend_mappings():
    definitions = {"EXISTING": 1}
    ConfigPlatformFeature("platform").add_platform_defs("mlif", definitions)
    assert definitions == {"EXISTING": 1, "PLATFORM": "mlif"}

    config = {}
    ConfigRunFeature("run").add_run_config(config)
    assert config == {"run.enabled": True}


class CacheFeature(SetupFeature):
    def get_required_cache_flags(self):
        return {"compiler": ["debug", "shared"], "runtime": ["debug"]}


def test_setup_feature_merges_cache_flags_without_duplicates():
    required = {"compiler": ["debug", "llvm"]}
    CacheFeature("cache").add_required_cache_flags(required)
    assert set(required["compiler"]) == {"debug", "shared", "llvm"}
    assert required["runtime"] == ["debug"]
