import pytest

# from mlonmcu.artifact import ArtifactFormat
from mlonmcu.artifact import lookup_artifacts

from mlonmcu.testing.helpers import (
    DEFAULT_BACKENDS,
    DEFAULT_TFLITE_MODELS,
    DEFAULT_MLIF_TARGETS,
    DEFAULT_ESPIDF_TARGETS,
    _test_run_platform,
    _test_compile_platform,
)


@pytest.mark.slow
@pytest.mark.user_context
@pytest.mark.parametrize("model_name", DEFAULT_TFLITE_MODELS)
@pytest.mark.parametrize("backend_name", DEFAULT_BACKENDS)
@pytest.mark.parametrize("target_name", DEFAULT_MLIF_TARGETS)
@pytest.mark.parametrize("feature_names", [[]])
@pytest.mark.parametrize("config", [{}])
def test_target_mlif(user_context, model_name, backend_name, target_name, models_dir, feature_names, config):
    _, artifacts = _test_run_platform(
        "mlif", backend_name, target_name, user_context, model_name, models_dir, feature_names, config
    )

    assert len(lookup_artifacts(artifacts, name="generic_mlonmcu")) == 1
    assert len(lookup_artifacts(artifacts, name="mlif_out.log")) == 1
    # TODO: check artifacts


@pytest.mark.slow
@pytest.mark.user_context
@pytest.mark.parametrize("model_name", DEFAULT_TFLITE_MODELS)
@pytest.mark.parametrize("backend_name", DEFAULT_BACKENDS)
@pytest.mark.parametrize("target_name", DEFAULT_ESPIDF_TARGETS)
@pytest.mark.parametrize("feature_names", [[]])
@pytest.mark.parametrize(
    "config", [{"espidf.wait_for_user": False, "tflmi.arena_size": 2**17, "tvmaot.arena_size": 2**17}]
)
def test_platform_espidf(user_context, model_name, backend_name, target_name, models_dir, feature_names, config):
    _, artifacts = _test_compile_platform(
        "espidf", backend_name, target_name, user_context, model_name, models_dir, feature_names, config
    )

    assert len(lookup_artifacts(artifacts, name="generic_mlonmcu")) == 1
    assert len(lookup_artifacts(artifacts, name="espidf_out.log")) == 1
    # TODO: check artifacts


@pytest.mark.slow
@pytest.mark.hardware
@pytest.mark.user_context
@pytest.mark.parametrize("model_name", DEFAULT_TFLITE_MODELS)
@pytest.mark.parametrize("backend_name", DEFAULT_BACKENDS)
@pytest.mark.parametrize("target_name", ["esp32c3"])
@pytest.mark.parametrize("feature_names", [[]])
@pytest.mark.parametrize(
    "config",
    [
        {
            "espidf.wait_for_user": False,
            "espidf.use_idf_monitor": False,
            "espidf.port": "/dev/ttyUSB0",
            "tflmi.arena_size": 2**17,  # esp32c3 ram ~300kB
            "tvmaot.arena_size": 2**17,
        }
    ],
)
def test_target_espidf(user_context, model_name, backend_name, target_name, models_dir, feature_names, config):
    _, artifacts = _test_run_platform(
        "espidf", backend_name, target_name, user_context, model_name, models_dir, feature_names, config
    )

    assert len(lookup_artifacts(artifacts, name="generic_mlonmcu")) == 1
    assert len(lookup_artifacts(artifacts, name="espidf_out.log")) == 1


@pytest.mark.slow
@pytest.mark.hardware
@pytest.mark.user_context
@pytest.mark.parametrize("model_name", DEFAULT_TFLITE_MODELS)
@pytest.mark.parametrize("backend_name", ["tvmllvm"])
@pytest.mark.parametrize("target_name", ["tvm_cpu"])
@pytest.mark.parametrize("feature_names", [[]])
@pytest.mark.parametrize(
    "config",
    [{}],
)
def test_target_tvm(user_context, model_name, backend_name, target_name, models_dir, feature_names, config):
    _, artifacts = _test_run_platform(
        "tvm", backend_name, target_name, user_context, model_name, models_dir, feature_names, config
    )

    assert len(lookup_artifacts(artifacts, name="tvm_cpu_out.log")) == 1
