from mlonmcu.environment.config import PathConfig
from mlonmcu.session.run import RunStage

# from mlonmcu.artifact import ArtifactFormat
from mlonmcu.artifact import lookup_artifacts

from mlonmcu.testing.helpers import _test_run_platform

import pytest


# TODO: add user_session fixture which handles cleanup via session.discard()


# Frontends

# TODO: make sure that we use quant/float models and several different operators
DEFAULT_TFLITE_MODELS = ["sine_model"]
DEFAULT_ONNX_MODELS = ["onnx_mnist"]
DEFAULT_RELAY_MODELS = ["test_cnn"]
DEFAULT_PB_MODELS = ["mobilenet_v1_1_0_224"]
ALL_PB_MODELS = ["mobilenet_v1_1_0_224", "mobilenet_v1_1_0_224_quant", "mobilenet_v1_0_25_128"]
DEFAULT_PADDLE_MODELS = ["paddle_resnet50"]

MODEL_FRONTENDS = {
    "sine_model": "tflite",
    "aww": "tflite",
    "onnx_mnist": "onnx",
    "test_cnn": "relay",
    "mobilenet_v1_1.0_224_frozen": "pb",  # or tflite
    "mobilenet_v1_1.0_224_quant": "pb",  # or tflite
    "mobilenet_v1_0.25_128": "pb",  # or tflite
    "paddle_resnet50": "paddle",
}

TARGET_PLATFORMS = {
    "host_x86": "mlif",
    "corstone300": "mlif",
    "etiss": "mlif",
    "spike": "mlif",
    "ovpsim": "mlif",
    "riscv_qemu": "mlif",
    "tvm_cpu": "tvm",
    "microtvm_host": "microtvm",
    "microtvm_spike": "microtvm",
    "microtvm_espidf": "microtvm",
    "microtvm_etiss": "microtvm",
    "microtvm_zephyr": "microtvm",
    "microtvm_arduino": "microtvm",
}

DEFAULT_FRONTENDS = ["tflite"]  # TODO: needs to match with the DEFAULT_MODELS
ALL_FRONTENDS = ["tflite", "pb", "relay", "onnx"]

DEFAULT_BACKENDS = ["tflmi", "tvmaot"]  # TODO: how about tvmrt/tflmc
ALL_BACKENDS = ["tflmi", "tflmc", "tvmaot", "tvmaotplus", "tvmrt", "tvmcg"]

DEFAULT_COMPILE_PLATFORMS = ["mlif", "espidf"]
ALL_COMPILE_PLATFORMS = ["mlif", "espidf", "zephyr"]

# DEFAULT_MLIF_TARGETS = ["host_x86", "etiss", "spike", "ovpsim", "corstone300"]
DEFAULT_MLIF_TARGETS = ["host_x86", "etiss", "spike", "corstone300"]
ALL_MLIF_TARGETS = ["host_x86", "etiss", "spike", "ovpsim", "corstone300", "riscv_qemu"]

DEFAULT_ESPIDF_TARGETS = ["esp32", "esp32c3"]
# ALL_ESPIDF_TARGETS = [] # TODO
DEFAULT_TARGETS = DEFAULT_MLIF_TARGETS + DEFAULT_ESPIDF_TARGETS

VEXT_TARGETS = ["spike", "ovpsim", "riscv_qemu"]  # TODO: etiss
# Spike PEXT support (0.92) is outdated compared to ovpsim (0.96)
PEXT_TARGETS = ["ovpsim"]  # TODO: etiss, spike

# RISCV_TARGETS = ["spike", "etiss", "ovpsim"]
RISCV_TARGETS = ["spike", "etiss", "ovpsim", "riscv_qemu"]

ARM_MVEI_TARGETS = ["corstone300"]
ARM_DSP_TARGETS = ["corstone300"]

DEBUG_ARENA_BACKENDS = ["tflmi", "tvmaot", "tvmaotplus", "tvmrt"]


def get_tvm_example_config(backend):
    return {
        f"{backend}.extra_pass_config": {"relay.FuseOps.max_depth": 0},  # TODO
        f"{backend}.disabled_passes": ["AlterOpLayout"],
        f"{backend}.target_device": "arm_cpu",
        f"{backend}.opt_level": 2,
    }


def _check_features(user_context, feature_names):
    for feature in feature_names:
        if not user_context.environment.has_feature(feature):
            pytest.skip(f"Feature '{feature}' is not enabled.")


def _init_run(user_context, models_dir, config):
    user_context.environment.paths["models"] = [PathConfig(models_dir)]
    session = user_context.create_session()
    return session, session.create_run(config=config)


def _test_frontend(frontend_name, user_context, model_name, models_dir, feature_names, config):
    user_config = user_context.environment.vars.copy()
    user_config.update(config)
    if not user_context.environment.has_frontend(frontend_name):
        pytest.skip(f"Frontend '{frontend_name}' is not enabled.")
    _check_features(user_context, feature_names)
    session, run = _init_run(user_context, models_dir, user_config)
    run.add_features_by_name(feature_names, context=user_context)
    run.add_frontend_by_name(frontend_name, context=user_context)
    run.add_model_by_name(model_name, context=user_context)
    assert session.process_runs(until=RunStage.LOAD, context=user_context)
    report = session.get_reports()
    df, artifacts = report.df, run.artifacts

    assert len(df) == 1
    assert df["Model"][0] == model_name
    assert df["Frontend"][0] == frontend_name

    assert len(lookup_artifacts(artifacts)) > 0
    return df, artifacts

    # TODO: test for metadata
    # TODO: test model data.c (after refactor)
    # artifacts.append(Artifact(f"{name}.{ext}", raw=raw, fmt=ArtifactFormat.RAW))
    # data_artifact = Artifact("data.c", content=data_src, fmt=ArtifactFormat.SOURCE)


@pytest.mark.user_context
@pytest.mark.parametrize("model_name", DEFAULT_TFLITE_MODELS)
@pytest.mark.parametrize("feature_names", [[], ["tflite_visualize"]])
@pytest.mark.parametrize("config", [{}])
def test_frontend_tflite(user_context, model_name, models_dir, feature_names, config):
    _, artifacts = _test_frontend("tflite", user_context, model_name, models_dir, feature_names, config)


@pytest.mark.user_context
@pytest.mark.parametrize("model_name", DEFAULT_TFLITE_MODELS)
# @pytest.mark.parametrize("feature_names", [[], ["tflite_visualize"]])
@pytest.mark.parametrize("feature_names", [["visualize"]])  # TODO: rename
@pytest.mark.parametrize("config", [{}])
def test_feature_tflite_visualize(user_context, model_name, models_dir, feature_names, config):
    _, artifacts = _test_frontend("tflite", user_context, model_name, models_dir, feature_names, config)
    assert len(lookup_artifacts(artifacts, name="tflite_visualize.html")) == 1


@pytest.mark.user_context
@pytest.mark.parametrize("model_name", DEFAULT_ONNX_MODELS)
@pytest.mark.parametrize("feature_names", [[]])
@pytest.mark.parametrize("config", [{}])
def test_frontend_onnx(user_context, model_name, models_dir, feature_names, config):
    _test_frontend("onnx", user_context, model_name, models_dir, feature_names, config)


@pytest.mark.user_context
@pytest.mark.parametrize("model_name", DEFAULT_RELAY_MODELS)
@pytest.mark.parametrize("feature_names", [[]])
@pytest.mark.parametrize("config", [{}])
def test_frontend_relay(user_context, model_name, models_dir, feature_names, config):
    _test_frontend("relay", user_context, model_name, models_dir, feature_names, config)


@pytest.mark.user_context
@pytest.mark.parametrize("model_name", DEFAULT_RELAY_MODELS)
@pytest.mark.parametrize("feature_names", [["relayviz"]])
@pytest.mark.parametrize("config", [{"relayviz.plotter": "term"}, {"relayviz.plotter": "dot"}])
def test_feature_relayviz(user_context, model_name, models_dir, feature_names, config):
    _, artifacts = _test_frontend("relay", user_context, model_name, models_dir, feature_names, config)
    ext = "txt" if config["relayviz.plotter"] == "term" else "pdf"
    assert len(lookup_artifacts(artifacts, name=f"relayviz.{ext}")) == 1


@pytest.mark.user_context
@pytest.mark.parametrize("model_name", DEFAULT_PB_MODELS)
@pytest.mark.parametrize("feature_names", [[]])
@pytest.mark.parametrize("config", [{}])
def test_frontend_pb(user_context, model_name, models_dir, feature_names, config):
    _test_frontend("pb", user_context, model_name, models_dir, feature_names, config)


@pytest.mark.skip(reason="unimplemented")
@pytest.mark.user_context
@pytest.mark.parametrize("model_name", DEFAULT_PADDLE_MODELS)
@pytest.mark.parametrize("feature_names", [[]])
@pytest.mark.parametrize("config", [{}])
def test_frontend_paddle(user_context, model_name, models_dir, feature_names, config):
    _test_frontend("paddle", user_context, model_name, models_dir, feature_names, config)


# Backends


# TODO: decide if execute on a per-framework basis?
def _test_backend(backend_name, user_context, model_name, models_dir, feature_names, config):
    user_config = user_context.environment.vars.copy()
    user_config.update(config)
    frontend_name = MODEL_FRONTENDS[model_name]
    if not user_context.environment.has_frontend(frontend_name):
        pytest.skip(f"Frontend '{frontend_name}' is not enabled.")
    if not user_context.environment.has_backend(backend_name):
        pytest.skip(f"Backend '{backend_name}' is not enabled.")
    _check_features(user_context, feature_names)
    session, run = _init_run(user_context, models_dir, user_config)
    run.add_features_by_name(feature_names, context=user_context)
    run.add_frontend_by_name(frontend_name, context=user_context)
    run.add_model_by_name(model_name, context=user_context)
    run.add_backend_by_name(backend_name, context=user_context)  # TODO: implicit Framework
    assert session.process_runs(until=RunStage.BUILD, context=user_context)
    report = session.get_reports()
    df, artifacts = report.df, run.artifacts

    assert len(df) == 1
    assert df["Model"][0] == model_name
    assert df["Backend"][0] == backend_name
    return df, artifacts


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
