# from mlonmcu.artifact import ArtifactFormat
from mlonmcu.artifact import lookup_artifacts

from mlonmcu.testing.helpers import _test_backend, DEFAULT_TFLITE_MODELS

import pytest


def get_tvm_example_config(backend):
    return {
        f"{backend}.extra_pass_config": {"relay.FuseOps.max_depth": 0},  # TODO
        f"{backend}.disabled_passes": ["AlterOpLayout"],
        f"{backend}.target_device": "arm_cpu",
        f"{backend}.opt_level": 2,
    }


@pytest.mark.user_context
@pytest.mark.parametrize("model_name", DEFAULT_TFLITE_MODELS)
@pytest.mark.parametrize("feature_names", [[]])
@pytest.mark.parametrize(
    "config", [{}, {"tflmi.arena_size": 2**20, "tflmi.ops": ["TODO"]}]  # TODO
)  # TODO: user should be ablte to overwrite sesstings parsed by frontend
def test_backend_tflmi(user_context, model_name, models_dir, feature_names, config):
    df, artifacts = _test_backend("tflmi", user_context, model_name, models_dir, feature_names, config)
    assert df["Framework"][0] == "tflm"

    assert len(lookup_artifacts(artifacts, name="model.cc")) == 1
    assert len(lookup_artifacts(artifacts, name="model.cc.h")) == 1
    assert len(lookup_artifacts(artifacts, name="tflmi_arena_size.txt")) == 1
    # TODO: check if non-empty
    # TODO: check if arena size is propagated


@pytest.mark.user_context
@pytest.mark.parametrize("model_name", DEFAULT_TFLITE_MODELS)
@pytest.mark.parametrize("feature_names", [[]])
@pytest.mark.parametrize("config", [{}])
def test_backend_tflmc(user_context, model_name, models_dir, feature_names, config):
    df, artifacts = _test_backend("tflmc", user_context, model_name, models_dir, feature_names, config)
    assert df["Framework"][0] == "tflm"

    assert len(lookup_artifacts(artifacts, name="model.cc")) == 1
    assert len(lookup_artifacts(artifacts, name="model.cc.h")) == 1
    assert len(lookup_artifacts(artifacts, name="tflmc_out.log")) == 1


@pytest.mark.slow
@pytest.mark.user_context
# @pytest.mark.parametrize("model_name", DEFAULT_TFLITE_MODELS + DEFAULT_RELAY_MODELS
# + DEFAULT_PB_MODELS + DEFAULT_ONNX_MODELS + DEFAULT_PADDLE_MODELS)
@pytest.mark.parametrize("model_name", DEFAULT_TFLITE_MODELS)
@pytest.mark.parametrize("feature_names", [[]])
@pytest.mark.parametrize(
    "config",
    [
        {},
        {
            **get_tvm_example_config("tvmaot"),
            "tvmaot.arena_size": 2**20,
            "tvmaot.alignment_bytes": 16,
        },
    ],
)
def test_backend_tvmaot(user_context, model_name, models_dir, feature_names, config):
    df, artifacts = _test_backend("tvmaot", user_context, model_name, models_dir, feature_names, config)
    assert df["Framework"][0] == "tvm"

    assert len(lookup_artifacts(artifacts, name="default.tar")) == 1
    assert len(lookup_artifacts(artifacts, name="tvmc_compile_out.log")) == 1
    # TODO: check if non-empty
    # TODO: check if arena/alignment updated


@pytest.mark.slow
@pytest.mark.user_context
@pytest.mark.parametrize("model_name", DEFAULT_TFLITE_MODELS)
@pytest.mark.parametrize("feature_names", [[]])
@pytest.mark.parametrize(
    "config",
    [
        {},
        {
            **get_tvm_example_config("tvmaotplus"),
            "tvmaotplus.arena_size": 2**20,
            "tvmaotplus.alignment_bytes": 16,
        },
    ],
)
def test_backend_tvmaotplus(user_context, model_name, models_dir, feature_names, config):
    df, artifacts = _test_backend("tvmaotplus", user_context, model_name, models_dir, feature_names, config)
    assert df["Framework"][0] == "tvm"

    assert len(lookup_artifacts(artifacts, name="default.tar")) == 1
    assert len(lookup_artifacts(artifacts, name="tvmc_compile_out.log")) == 1
    # TODO: check if non-empty
    # TODO: check if arena/alignment updated


@pytest.mark.slow
@pytest.mark.user_context
@pytest.mark.parametrize("model_name", DEFAULT_TFLITE_MODELS)
@pytest.mark.parametrize("feature_names", [[]])
@pytest.mark.parametrize(
    "config",
    [
        {},
        {
            **get_tvm_example_config("tvmrt"),
            "tvmrt.arena_size": 2**20,
        },
    ],
)  # TODO: combine tvm common configs
def test_backend_tvmrt(user_context, model_name, models_dir, feature_names, config):
    df, artifacts = _test_backend("tvmrt", user_context, model_name, models_dir, feature_names, config)
    assert df["Framework"][0] == "tvm"

    assert len(lookup_artifacts(artifacts, name="default.tar")) == 1
    assert len(lookup_artifacts(artifacts, name="tvmc_compile_out.log")) == 1
    # TODO: check if non-empty
    # TODO: check if arena updated


@pytest.mark.slow
@pytest.mark.user_context
@pytest.mark.parametrize("model_name", DEFAULT_TFLITE_MODELS)
@pytest.mark.parametrize("feature_names", [[]])
@pytest.mark.parametrize(
    "config",
    [
        {},
        {
            **get_tvm_example_config("tvmcg"),
            "arena_size": 2**20,
        },
    ],
)
def test_backend_tvmcg(user_context, model_name, models_dir, feature_names, config):
    df, artifacts = _test_backend("tvmcg", user_context, model_name, models_dir, feature_names, config)
    assert df["Framework"][0] == "tvm"

    assert len(lookup_artifacts(artifacts, name="default.tar")) == 1
    assert len(lookup_artifacts(artifacts, name="tvmc_compile_out.log")) == 1
    # TODO: check if non-empty
    # TODO: check if arena updated
