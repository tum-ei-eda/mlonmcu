from mlonmcu.environment.config import PathConfig
from mlonmcu.session.run import RunStage

# from mlonmcu.artifact import ArtifactFormat
from mlonmcu.artifact import lookup_artifacts

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
    "etiss_pulpino": "mlif",
    "spike": "mlif",
    "ovpsim": "mlif",
    "riscv_qemu": "mlif",
    "tvm_cpu": "tvm",
    "microtvm_host": "microtvm",
    "microtvm_spike": "microtvm",
    "microtvm_espidf": "microtvm",
    "microtvm_etissvp": "microtvm",
    "microtvm_zephyr": "microtvm",
    "microtvm_arduino": "microtvm",
}

DEFAULT_FRONTENDS = ["tflite"]  # TODO: needs to match with the DEFAULT_MODELS
ALL_FRONTENDS = ["tflite", "pb", "relay", "onnx"]

DEFAULT_BACKENDS = ["tflmi", "tvmaot"]  # TODO: how about tvmrt/tflmc
ALL_BACKENDS = ["tflmi", "tflmc", "tvmaot", "tvmaotplus", "tvmrt", "tvmcg"]

DEFAULT_COMPILE_PLATFORMS = ["mlif", "espidf"]
ALL_COMPILE_PLATFORMS = ["mlif", "espidf", "zephyr"]

# DEFAULT_MLIF_TARGETS = ["host_x86", "etiss_pulpino", "spike", "ovpsim", "corstone300"]
DEFAULT_MLIF_TARGETS = ["host_x86", "etiss_pulpino", "spike", "corstone300"]
ALL_MLIF_TARGETS = ["host_x86", "etiss_pulpino", "spike", "ovpsim", "corstone300", "riscv_qemu"]

DEFAULT_ESPIDF_TARGETS = ["esp32", "esp32c3"]
# ALL_ESPIDF_TARGETS = [] # TODO
DEFAULT_TARGETS = DEFAULT_MLIF_TARGETS + DEFAULT_ESPIDF_TARGETS

VEXT_TARGETS = ["spike", "ovpsim", "riscv_qemu"]  # TODO: etiss
# Spike PEXT support (0.92) is outdated compared to ovpsim (0.96)
PEXT_TARGETS = ["ovpsim"]  # TODO: etiss, spike

# RISCV_TARGETS = ["spike", "etiss_pulpino", "ovpsim"]
RISCV_TARGETS = ["spike", "etiss_pulpino", "ovpsim", "riscv_qemu"]

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


    # TODO: check if non-empty


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


# Platforms(Compile)/Targets(Run)


def _test_compile_platform(
    platform_name, backend_name, target_name, user_context, model_name, models_dir, feature_names, config
):
    user_config = user_context.environment.vars.copy()
    user_config.update(config)
    frontend_name = MODEL_FRONTENDS[model_name]
    if platform_name is None:
        platform_name = TARGET_PLATFORMS[target_name]
    if not user_context.environment.has_frontend(frontend_name):
        pytest.skip(f"Frontend '{frontend_name}' is not enabled.")
    if not user_context.environment.has_backend(backend_name):
        pytest.skip(f"Backend '{backend_name}' is not enabled.")
    if not user_context.environment.has_platform(platform_name):
        pytest.skip(f"Platform '{platform_name}' is not enabled.")  # TODO: not enabled -> not installed
    if not user_context.environment.has_target(target_name):
        pytest.skip(f"Target '{target_name}' is not enabled.")  # TODO: remove check?
    _check_features(user_context, feature_names)
    session, run = _init_run(user_context, models_dir, user_config)
    run.add_features_by_name(feature_names, context=user_context)
    run.add_frontend_by_name(frontend_name, context=user_context)
    run.add_model_by_name(model_name, context=user_context)
    run.add_backend_by_name(backend_name, context=user_context)  # TODO: implicit Framework
    run.add_platform_by_name(platform_name, context=user_context)
    run.add_target_by_name(target_name, context=user_context)
    assert session.process_runs(until=RunStage.COMPILE, context=user_context)
    report = session.get_reports()
    df, artifacts = report.df, run.artifacts

    assert len(df) == 1
    assert df["Model"][0] == model_name
    assert df["Platform"][0] == platform_name
    assert df["Target"][0] == target_name
    return df, artifacts


def _test_run_platform(
    platform_name, backend_name, target_name, user_context, model_name, models_dir, feature_names, config
):
    user_config = user_context.environment.vars.copy()
    user_config.update(config)
    frontend_name = MODEL_FRONTENDS[model_name]
    if platform_name is None:
        platform_name = TARGET_PLATFORMS[target_name]
    if not user_context.environment.has_frontend(frontend_name):
        pytest.skip(f"Frontend '{frontend_name}' is not enabled.")
    if not user_context.environment.has_backend(backend_name):
        pytest.skip(f"Backend '{backend_name}' is not enabled.")
    if not user_context.environment.has_platform(platform_name):
        pytest.skip(f"Platform '{platform_name}' is not enabled.")  # TODO: not enabled -> not installed
    if not user_context.environment.has_target(target_name):
        pytest.skip(f"Target '{target_name}' is not enabled.")  # TODO: remove check?
    _check_features(user_context, feature_names)
    session, run = _init_run(user_context, models_dir, user_config)
    run.add_features_by_name(feature_names, context=user_context)
    run.add_frontend_by_name(frontend_name, context=user_context)
    run.add_model_by_name(model_name, context=user_context)
    run.add_backend_by_name(backend_name, context=user_context)  # TODO: implicit Framework
    run.add_platform_by_name(platform_name, context=user_context)
    run.add_target_by_name(target_name, context=user_context)
    assert session.process_runs(until=RunStage.RUN, context=user_context)
    report = session.get_reports()
    df, artifacts = report.df, run.artifacts

    assert len(df) == 1
    assert df["Model"][0] == model_name
    assert df["Platform"][0] == platform_name
    assert df["Target"][0] == target_name
    return df, artifacts


@pytest.mark.slow
@pytest.mark.user_context
@pytest.mark.parametrize("model_name", DEFAULT_TFLITE_MODELS)
@pytest.mark.parametrize("backend_name", DEFAULT_BACKENDS)
@pytest.mark.parametrize("target_name", DEFAULT_MLIF_TARGETS)
@pytest.mark.parametrize("feature_names", [[]])
@pytest.mark.parametrize("config", [{}])
def test_platform_mlif(user_context, model_name, backend_name, target_name, models_dir, feature_names, config):
    _, artifacts = _test_compile_platform(
        "mlif", backend_name, target_name, user_context, model_name, models_dir, feature_names, config
    )

    assert len(lookup_artifacts(artifacts, name="generic_mlif")) == 1
    assert len(lookup_artifacts(artifacts, name="mlif_out.log")) == 1
    # TODO: check artifacts


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

    assert len(lookup_artifacts(artifacts, name="generic_mlif")) == 1
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

    assert len(lookup_artifacts(artifacts, name="generic_mlif")) == 1
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

    assert len(lookup_artifacts(artifacts, name="generic_mlif")) == 1
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

    assert len(lookup_artifacts(artifacts, name="tvmc_run_out.log")) == 1


# TODO: test microtvm platforn and targets - Needs zephyr or arduino installed -> Manually or via mlonmcu?

# # PostProcesses


# Features
@pytest.mark.slow
@pytest.mark.user_context
@pytest.mark.parametrize("model_name", DEFAULT_TFLITE_MODELS)  # TODO: single model would be enough
@pytest.mark.parametrize("backend_name", DEBUG_ARENA_BACKENDS)
@pytest.mark.parametrize("target_name", ["host_x86"])
@pytest.mark.parametrize("platform_name", ["mlif"])
@pytest.mark.parametrize(
    "feature_names", [["debug_arena", "debug"]]
)  # TODO: should debug_arena set {target}.print_outputs automatically?
@pytest.mark.parametrize("config", [{}])
def test_feature_debug_arena(
    user_context, model_name, backend_name, target_name, platform_name, models_dir, feature_names, config
):
    df, artifacts = _test_run_platform(
        "mlif", backend_name, target_name, user_context, model_name, models_dir, feature_names, config
    )

    assert len(lookup_artifacts(artifacts, name="host_x86_out.log")) == 1
    assert "debug_arena" in df["Features"][0]
    # Check generated code
    # Check stdout


@pytest.mark.slow
@pytest.mark.user_context
@pytest.mark.parametrize(
    # "model_name", DEFAULT_TFLITE_MODELS
    "model_name",
    ["aww"],
)  # TODO: add test model for this, also test with wrong data
@pytest.mark.parametrize("backend_name", ["tflmi", "tvmaot"])
@pytest.mark.parametrize("target_name", ["etiss_pulpino"])
@pytest.mark.parametrize("feature_names", [["validate", "debug"]])  # currently validate does not imply debug
@pytest.mark.parametrize("config", [{}])
def test_feature_validate(user_context, model_name, backend_name, target_name, models_dir, feature_names, config):
    df, artifacts = _test_run_platform(
        None, backend_name, target_name, user_context, model_name, models_dir, feature_names, config
    )

    assert len(lookup_artifacts(artifacts, name=f"{target_name}_out.log")) == 1
    # Check generated code
    assert "validate" in df["Features"][0]
    assert "Validation" in df.columns
    assert df["Validation"][0]  # if model has validation data else, missing/NaN/None?
    # TODO: force correct
    # TODO: force missmatch
    # TODO: force missing
    # TODO: force invalid size


@pytest.mark.slow
@pytest.mark.user_context
@pytest.mark.parametrize(
    "model_name", DEFAULT_TFLITE_MODELS
)  # TODO: add test model for this, also test with wrong data
@pytest.mark.parametrize("backend_name", ["tflmi", "tvmaot"])
@pytest.mark.parametrize("target_name", ["host_x86"])
@pytest.mark.parametrize("feature_names", [["debug"]])  # currently validate does not imply debug
@pytest.mark.parametrize("config", [{}])
def test_feature_debug(user_context, model_name, backend_name, target_name, models_dir, feature_names, config):
    df, artifacts = _test_run_platform(
        None, backend_name, target_name, user_context, model_name, models_dir, feature_names, config
    )
    assert "debug" in df["Features"][0]
    # TODO: stdout with test model
    # TODO: 2 runs to compare ROM/RAM/Cycles?


@pytest.mark.slow
@pytest.mark.user_context
@pytest.mark.parametrize(
    "model_name", DEFAULT_TFLITE_MODELS
)  # TODO: add test model for this, also test with wrong data
@pytest.mark.parametrize("backend_name", ["tflmi"])
@pytest.mark.parametrize("target_name", DEFAULT_MLIF_TARGETS)
@pytest.mark.parametrize(
    "feature_names", [["muriscvnn"], ["muriscvnn", "debug"]]
)  # currently validate does not imply debug
@pytest.mark.parametrize("config", [{}])
def test_feature_muriscvnn(user_context, model_name, backend_name, target_name, models_dir, feature_names, config):
    df, artifacts = _test_run_platform(
        None, backend_name, target_name, user_context, model_name, models_dir, feature_names, config
    )
    assert "muriscvnn" in df["Features"][0]
    # TODO: find out if kernels are actually linked?
    # TODO: 2 runs to compare ROM/RAM/Cycles?


@pytest.mark.slow
@pytest.mark.user_context
@pytest.mark.parametrize(
    "model_name", DEFAULT_TFLITE_MODELS
)  # TODO: add test model for this, also test with wrong data
@pytest.mark.parametrize("frontend_name", DEFAULT_FRONTENDS)  # Validate is frontend feature as well
@pytest.mark.parametrize(
    "backend_name", ["tflmi"]  # -> add tvm if we have a test model for this
)  # TODO: Single backend would be fine, but it has to be enabled
@pytest.mark.parametrize("target_name", DEFAULT_MLIF_TARGETS)  # TODO: more targets (without vext)
@pytest.mark.parametrize(
    "platform_name", ["mlif"]
)  # If we would rename host_x86 to linux we could also use espidf here?
@pytest.mark.parametrize(
    "feature_names", [["cmsisnn"], ["cmsisnn", "debug"]]
)  # currently validate does not imply debug
@pytest.mark.parametrize("config", [{}])
def test_feature_cmsisnn(
    user_context, frontend_name, model_name, backend_name, target_name, platform_name, models_dir, feature_names, config
):
    if not user_context.environment.has_frontend(frontend_name):
        pytest.skip(f"Frontend '{frontend_name}' is not enabled.")
    if not user_context.environment.has_backend(backend_name):
        pytest.skip(f"Backend '{backend_name}' is not enabled.")
    if not user_context.environment.has_platform(platform_name):
        pytest.skip(f"Platform '{platform_name}' is not enabled.")  # TODO: not enabled -> not installed
    if not user_context.environment.has_target(target_name):
        pytest.skip(f"Target '{target_name}' is not enabled.")  # TODO: not enabled -> not installed
    for feature in feature_names:
        if not user_context.environment.has_feature(feature):
            pytest.skip(f"Feature '{feature}' is not enabled.")
    user_context.environment.paths["models"] = [PathConfig(models_dir)]
    session = user_context.create_session()
    run = session.create_run(config=config)
    run.add_features_by_name(feature_names, context=user_context)
    run.add_frontend_by_name(frontend_name, context=user_context)
    run.add_model_by_name(model_name, context=user_context)
    run.add_backend_by_name(backend_name, context=user_context)
    run.add_platform_by_name(platform_name, context=user_context)
    run.add_target_by_name(target_name, context=user_context)
    success = session.process_runs(until=RunStage.RUN, context=user_context)
    report = session.get_reports()
    df = report.df
    assert success
    assert len(df) == 1
    assert "cmsisnn" in df["Features"][0]
    # TODO: find out if kernels are actually linked?
    # TODO: 2 runs to compare ROM/RAM/Cycles?


@pytest.mark.slow
@pytest.mark.user_context
@pytest.mark.parametrize(
    "model_name", DEFAULT_TFLITE_MODELS
)  # TODO: add test model for this, also test with wrong data
@pytest.mark.parametrize("frontend_name", DEFAULT_FRONTENDS)  # Validate is frontend feature as well
@pytest.mark.parametrize(
    "backend_name", ["tvmaot"]  # -> add tvm if we have a test model for this
)  # TODO: Single backend would be fine, but it has to be enabled
@pytest.mark.parametrize("target_name", DEFAULT_MLIF_TARGETS)
@pytest.mark.parametrize(
    "platform_name", ["mlif"]
)  # If we would rename host_x86 to linux we could also use espidf here?
@pytest.mark.parametrize(
    "feature_names", [["muriscvnnbyoc"], ["muriscvnnbyoc", "debug"]]
)  # currently validate does not imply debug
@pytest.mark.parametrize("config", [{}])
def test_feature_muriscvnnbyoc(
    user_context, frontend_name, model_name, backend_name, target_name, platform_name, models_dir, feature_names, config
):
    if not user_context.environment.has_frontend(frontend_name):
        pytest.skip(f"Frontend '{frontend_name}' is not enabled.")
    if not user_context.environment.has_backend(backend_name):
        pytest.skip(f"Backend '{backend_name}' is not enabled.")
    if not user_context.environment.has_platform(platform_name):
        pytest.skip(f"Platform '{platform_name}' is not enabled.")  # TODO: not enabled -> not installed
    if not user_context.environment.has_target(target_name):
        pytest.skip(f"Target '{target_name}' is not enabled.")  # TODO: not enabled -> not installed
    for feature in feature_names:
        if not user_context.environment.has_feature(feature):
            pytest.skip(f"Feature '{feature}' is not enabled.")
    user_context.environment.paths["models"] = [PathConfig(models_dir)]
    session = user_context.create_session()
    run = session.create_run(config=config)
    run.add_features_by_name(feature_names, context=user_context)
    run.add_frontend_by_name(frontend_name, context=user_context)
    run.add_model_by_name(model_name, context=user_context)
    run.add_backend_by_name(backend_name, context=user_context)
    run.add_platform_by_name(platform_name, context=user_context)
    run.add_target_by_name(target_name, context=user_context)
    success = session.process_runs(until=RunStage.RUN, context=user_context)
    report = session.get_reports()
    df = report.df
    assert success
    assert len(df) == 1
    assert "muriscvnnbyoc" in df["Features"][0]
    # TODO: find out if kernels are actually linked?
    # TODO: 2 runs to compare ROM/RAM/Cycles?


@pytest.mark.slow
@pytest.mark.user_context
@pytest.mark.parametrize(
    "model_name", DEFAULT_TFLITE_MODELS
)  # TODO: add test model for this, also test with wrong data
@pytest.mark.parametrize("frontend_name", DEFAULT_FRONTENDS)  # Validate is frontend feature as well
@pytest.mark.parametrize(
    "backend_name", ["tvmaot"]  # -> add tvm if we have a test model for this
)  # TODO: Single backend would be fine, but it has to be enabled
@pytest.mark.parametrize("target_name", DEFAULT_MLIF_TARGETS)  # TODO: more targets (without vext)
@pytest.mark.parametrize(
    "platform_name", ["mlif"]
)  # If we would rename host_x86 to linux we could also use espidf here?
@pytest.mark.parametrize(
    "feature_names", [["cmsisnnbyoc"], ["cmsisnnbyoc", "debug"]]
)  # currently validate does not imply debug
@pytest.mark.parametrize("config", [{}])
def test_feature_cmsisnnbyoc(
    user_context, frontend_name, model_name, backend_name, target_name, platform_name, models_dir, feature_names, config
):
    if not user_context.environment.has_frontend(frontend_name):
        pytest.skip(f"Frontend '{frontend_name}' is not enabled.")
    if not user_context.environment.has_backend(backend_name):
        pytest.skip(f"Backend '{backend_name}' is not enabled.")
    if not user_context.environment.has_platform(platform_name):
        pytest.skip(f"Platform '{platform_name}' is not enabled.")  # TODO: not enabled -> not installed
    if not user_context.environment.has_target(target_name):
        pytest.skip(f"Target '{target_name}' is not enabled.")  # TODO: not enabled -> not installed
    for feature in feature_names:
        if not user_context.environment.has_feature(feature):
            pytest.skip(f"Feature '{feature}' is not enabled.")
    user_context.environment.paths["models"] = [PathConfig(models_dir)]
    session = user_context.create_session()
    run = session.create_run(config=config)
    run.add_features_by_name(feature_names, context=user_context)
    run.add_frontend_by_name(frontend_name, context=user_context)
    run.add_model_by_name(model_name, context=user_context)
    run.add_backend_by_name(backend_name, context=user_context)
    run.add_platform_by_name(platform_name, context=user_context)
    run.add_target_by_name(target_name, context=user_context)
    success = session.process_runs(until=RunStage.RUN, context=user_context)
    report = session.get_reports()
    df = report.df
    assert success
    assert len(df) == 1
    assert "cmsisnnbyoc" in df["Features"][0]
    # TODO: find out if kernels are actually linked?
    # TODO: 2 runs to compare ROM/RAM/Cycles?


@pytest.mark.slow
@pytest.mark.user_context
@pytest.mark.parametrize(
    "model_name", DEFAULT_TFLITE_MODELS
)  # TODO: add test model for this, also test with wrong data
@pytest.mark.parametrize("frontend_name", DEFAULT_FRONTENDS)  # Validate is frontend feature as well
@pytest.mark.parametrize(
    "backend_name", ["tflmi"]  # -> add tvm if we have a test model for this
)  # TODO: Single backend would be fine, but it has to be enabled
@pytest.mark.parametrize("target_name", VEXT_TARGETS)  # TODO: any backend would work for scalar code...
@pytest.mark.parametrize(
    "platform_name", ["mlif"]
)  # If we would rename host_x86 to linux we could also use espidf here?
@pytest.mark.parametrize("feature_names", [["vext"], ["vext", "muriscvnn"]])  # currently validate does not imply debug
@pytest.mark.parametrize("config", [{"vext.vlen": 128}])  # TODO: add multiple vlens
def test_feature_vext(
    user_context, frontend_name, model_name, backend_name, target_name, platform_name, models_dir, feature_names, config
):
    user_config = user_context.environment.vars.copy()
    user_config.update(config)
    if not user_context.environment.has_frontend(frontend_name):
        pytest.skip(f"Frontend '{frontend_name}' is not enabled.")
    if not user_context.environment.has_backend(backend_name):
        pytest.skip(f"Backend '{backend_name}' is not enabled.")
    if not user_context.environment.has_platform(platform_name):
        pytest.skip(f"Platform '{platform_name}' is not enabled.")  # TODO: not enabled -> not installed
    if not user_context.environment.has_target(target_name):
        pytest.skip(f"Target '{target_name}' is not enabled.")  # TODO: not enabled -> not installed
    for feature in feature_names:
        if not user_context.environment.has_feature(feature):
            pytest.skip(f"Feature '{feature}' is not enabled.")
    user_context.environment.paths["models"] = [PathConfig(models_dir)]
    session = user_context.create_session()
    run = session.create_run(config=user_config)
    run.add_features_by_name(feature_names, context=user_context)
    run.add_frontend_by_name(frontend_name, context=user_context)
    run.add_model_by_name(model_name, context=user_context)
    run.add_backend_by_name(backend_name, context=user_context)
    run.add_platform_by_name(platform_name, context=user_context)
    run.add_target_by_name(target_name, context=user_context)
    success = session.process_runs(until=RunStage.RUN, context=user_context)
    report = session.get_reports()
    df = report.df
    assert success
    assert len(df) == 1
    assert "vext" in df["Features"][0]
    # TODO: find out if kernels are actually linked?
    # TODO: 2 runs to compare ROM/RAM/Cycles?


@pytest.mark.slow
@pytest.mark.user_context
@pytest.mark.parametrize(
    "model_name", DEFAULT_TFLITE_MODELS
)  # TODO: add test model for this, also test with wrong data
@pytest.mark.parametrize("frontend_name", DEFAULT_FRONTENDS)  # Validate is frontend feature as well
@pytest.mark.parametrize(
    "backend_name", ["tflmi"]  # -> add tvm if we have a test model for this
)  # TODO: Single backend would be fine, but it has to be enabled
@pytest.mark.parametrize("target_name", PEXT_TARGETS)  # TODO: any backend would work for scalar code...
@pytest.mark.parametrize(
    "platform_name", ["mlif"]
)  # If we would rename host_x86 to linux we could also use espidf here?
@pytest.mark.parametrize("feature_names", [["pext"], ["pext", "muriscvnn"]])  # currently validate does not imply debug
@pytest.mark.parametrize("config", [{}])  # TODO: add multiple vlens
def test_feature_pext(
    user_context, frontend_name, model_name, backend_name, target_name, platform_name, models_dir, feature_names, config
):
    user_config = user_context.environment.vars.copy()
    user_config.update(config)
    if not user_context.environment.has_frontend(frontend_name):
        pytest.skip(f"Frontend '{frontend_name}' is not enabled.")
    if not user_context.environment.has_backend(backend_name):
        pytest.skip(f"Backend '{backend_name}' is not enabled.")
    if not user_context.environment.has_platform(platform_name):
        pytest.skip(f"Platform '{platform_name}' is not enabled.")  # TODO: not enabled -> not installed
    if not user_context.environment.has_target(target_name):
        pytest.skip(f"Target '{target_name}' is not enabled.")  # TODO: not enabled -> not installed
    for feature in feature_names:
        if not user_context.environment.has_feature(feature):
            pytest.skip(f"Feature '{feature}' is not enabled.")
    user_context.environment.paths["models"] = [PathConfig(models_dir)]
    session = user_context.create_session()
    run = session.create_run(config=user_config)
    run.add_features_by_name(feature_names, context=user_context)
    run.add_frontend_by_name(frontend_name, context=user_context)
    run.add_model_by_name(model_name, context=user_context)
    run.add_backend_by_name(backend_name, context=user_context)
    run.add_platform_by_name(platform_name, context=user_context)
    run.add_target_by_name(target_name, context=user_context)
    success = session.process_runs(until=RunStage.RUN, context=user_context)
    report = session.get_reports()
    df = report.df
    assert success
    assert len(df) == 1
    assert "pext" in df["Features"][0]
    # TODO: find out if kernels are actually linked?
    # TODO: 2 runs to compare ROM/RAM/Cycles?


@pytest.mark.slow
@pytest.mark.user_context
@pytest.mark.parametrize(
    "model_name", DEFAULT_TFLITE_MODELS
)  # TODO: add test model for this, also test with wrong data
@pytest.mark.parametrize("frontend_name", DEFAULT_FRONTENDS)  # Validate is frontend feature as well
@pytest.mark.parametrize(
    "backend_name", ["tflmi"]  # -> add tvm if we have a test model for this
)  # TODO: Single backend would be fine, but it has to be enabled
@pytest.mark.parametrize("target_name", ARM_MVEI_TARGETS)  # TODO: any backend would work for scalar code...
@pytest.mark.parametrize(
    "platform_name", ["mlif"]
)  # If we would rename host_x86 to linux we could also use espidf here?
@pytest.mark.parametrize(
    "feature_names", [["arm_mvei"], ["arm_mvei", "cmsisnn"]]
)  # currently validate does not imply debug
@pytest.mark.parametrize("config", [{}])  # TODO: add multiple vlens
def test_feature_arm_mvei(
    user_context, frontend_name, model_name, backend_name, target_name, platform_name, models_dir, feature_names, config
):
    user_config = user_context.environment.vars.copy()
    user_config.update(config)
    if not user_context.environment.has_frontend(frontend_name):
        pytest.skip(f"Frontend '{frontend_name}' is not enabled.")
    if not user_context.environment.has_backend(backend_name):
        pytest.skip(f"Backend '{backend_name}' is not enabled.")
    if not user_context.environment.has_platform(platform_name):
        pytest.skip(f"Platform '{platform_name}' is not enabled.")  # TODO: not enabled -> not installed
    if not user_context.environment.has_target(target_name):
        pytest.skip(f"Target '{target_name}' is not enabled.")  # TODO: not enabled -> not installed
    for feature in feature_names:
        if not user_context.environment.has_feature(feature):
            pytest.skip(f"Feature '{feature}' is not enabled.")
    user_context.environment.paths["models"] = [PathConfig(models_dir)]
    session = user_context.create_session()
    run = session.create_run(config=user_config)
    run.add_features_by_name(feature_names, context=user_context)
    run.add_frontend_by_name(frontend_name, context=user_context)
    run.add_model_by_name(model_name, context=user_context)
    run.add_backend_by_name(backend_name, context=user_context)
    run.add_platform_by_name(platform_name, context=user_context)
    run.add_target_by_name(target_name, context=user_context)
    success = session.process_runs(until=RunStage.RUN, context=user_context)
    report = session.get_reports()
    df = report.df
    assert success
    assert len(df) == 1
    assert "arm_mvei" in df["Features"][0]
    # TODO: find out if kernels are actually linked?
    # TODO: 2 runs to compare ROM/RAM/Cycles?


@pytest.mark.slow
@pytest.mark.user_context
@pytest.mark.parametrize(
    "model_name", DEFAULT_TFLITE_MODELS
)  # TODO: add test model for this, also test with wrong data
@pytest.mark.parametrize("frontend_name", DEFAULT_FRONTENDS)  # Validate is frontend feature as well
@pytest.mark.parametrize(
    "backend_name", ["tflmi"]  # -> add tvm if we have a test model for this
)  # TODO: Single backend would be fine, but it has to be enabled
@pytest.mark.parametrize("target_name", ARM_DSP_TARGETS)  # TODO: any backend would work for scalar code...
@pytest.mark.parametrize(
    "platform_name", ["mlif"]
)  # If we would rename host_x86 to linux we could also use espidf here?
@pytest.mark.parametrize(
    "feature_names", [["arm_dsp"], ["arm_dsp", "cmsisnn"]]
)  # currently validate does not imply debug
@pytest.mark.parametrize("config", [{}])  # TODO: add multiple vlens
def test_feature_arm_dsp(
    user_context, frontend_name, model_name, backend_name, target_name, platform_name, models_dir, feature_names, config
):
    user_config = user_context.environment.vars.copy()
    user_config.update(config)
    if not user_context.environment.has_frontend(frontend_name):
        pytest.skip(f"Frontend '{frontend_name}' is not enabled.")
    if not user_context.environment.has_backend(backend_name):
        pytest.skip(f"Backend '{backend_name}' is not enabled.")
    if not user_context.environment.has_platform(platform_name):
        pytest.skip(f"Platform '{platform_name}' is not enabled.")  # TODO: not enabled -> not installed
    if not user_context.environment.has_target(target_name):
        pytest.skip(f"Target '{target_name}' is not enabled.")  # TODO: not enabled -> not installed
    for feature in feature_names:
        if not user_context.environment.has_feature(feature):
            pytest.skip(f"Feature '{feature}' is not enabled.")
    user_context.environment.paths["models"] = [PathConfig(models_dir)]
    session = user_context.create_session()
    run = session.create_run(config=user_config)
    run.add_features_by_name(feature_names, context=user_context)
    run.add_frontend_by_name(frontend_name, context=user_context)
    run.add_model_by_name(model_name, context=user_context)
    run.add_backend_by_name(backend_name, context=user_context)
    run.add_platform_by_name(platform_name, context=user_context)
    run.add_target_by_name(target_name, context=user_context)
    success = session.process_runs(until=RunStage.RUN, context=user_context)
    report = session.get_reports()
    df = report.df
    assert success
    assert len(df) == 1
    assert "arm_dsp" in df["Features"][0]
    # TODO: find out if kernels are actually linked?
    # TODO: 2 runs to compare ROM/RAM/Cycles?


@pytest.mark.slow
@pytest.mark.user_context
@pytest.mark.parametrize(
    "model_name", DEFAULT_TFLITE_MODELS
)  # TODO: add test model for this, also test with wrong data
@pytest.mark.parametrize("frontend_name", DEFAULT_FRONTENDS)  # Validate is frontend feature as well
@pytest.mark.parametrize(
    "backend_name", ["tflmi"]  # -> add tvm if we have a test model for this
)  # TODO: Single backend would be fine, but it has to be enabled
@pytest.mark.parametrize("target_name", ["etiss_pulpino"])
@pytest.mark.parametrize(
    "platform_name", ["mlif"]
)  # If we would rename host_x86 to linux we could also use espidf here?
@pytest.mark.parametrize("feature_names", [["etissdbg"]])  # This is not etiss_pulpino.verbose=1!!!
@pytest.mark.parametrize("config", [{}])
def test_feature_etissdbg(
    user_context, frontend_name, model_name, backend_name, target_name, platform_name, models_dir, feature_names, config
):
    user_config = user_context.environment.vars.copy()
    user_config.update(config)
    if not user_context.environment.has_frontend(frontend_name):
        pytest.skip(f"Frontend '{frontend_name}' is not enabled.")
    if not user_context.environment.has_backend(backend_name):
        pytest.skip(f"Backend '{backend_name}' is not enabled.")
    if not user_context.environment.has_platform(platform_name):
        pytest.skip(f"Platform '{platform_name}' is not enabled.")  # TODO: not enabled -> not installed
    if not user_context.environment.has_target(target_name):
        pytest.skip(f"Target '{target_name}' is not enabled.")  # TODO: not enabled -> not installed
    for feature in feature_names:
        if not user_context.environment.has_feature(feature):
            pytest.skip(f"Feature '{feature}' is not enabled.")
    user_context.environment.paths["models"] = [PathConfig(models_dir)]
    session = user_context.create_session()
    run = session.create_run(config=user_config)
    run.add_features_by_name(feature_names, context=user_context)
    run.add_frontend_by_name(frontend_name, context=user_context)
    run.add_model_by_name(model_name, context=user_context)
    run.add_backend_by_name(backend_name, context=user_context)
    run.add_platform_by_name(platform_name, context=user_context)
    run.add_target_by_name(target_name, context=user_context)
    success = session.process_runs(until=RunStage.COMPILE, context=user_context)
    report = session.get_reports()
    df = report.df
    assert success
    assert len(df) == 1
    assert "etissdbg" in df["Features"][0]
    # TODO: run gdb but how?
    # TODO: check stdout


@pytest.mark.slow
@pytest.mark.user_context
@pytest.mark.parametrize(
    "model_name", DEFAULT_TFLITE_MODELS
)  # TODO: add test model for this, also test with wrong data
@pytest.mark.parametrize("frontend_name", DEFAULT_FRONTENDS)  # Validate is frontend feature as well
@pytest.mark.parametrize(
    "backend_name", ["tflmi"]  # -> add tvm if we have a test model for this
)  # TODO: Single backend would be fine, but it has to be enabled
@pytest.mark.parametrize("target_name", ["etiss_pulpino"])
@pytest.mark.parametrize(
    "platform_name", ["mlif"]
)  # If we would rename host_x86 to linux we could also use espidf here?
@pytest.mark.parametrize("feature_names", [["trace"]])  # currently validate does not imply debug
@pytest.mark.parametrize("config", [{}])
def test_feature_trace(
    user_context, frontend_name, model_name, backend_name, target_name, platform_name, models_dir, feature_names, config
):
    if not user_context.environment.has_frontend(frontend_name):
        pytest.skip(f"Frontend '{frontend_name}' is not enabled.")
    if not user_context.environment.has_backend(backend_name):
        pytest.skip(f"Backend '{backend_name}' is not enabled.")
    if not user_context.environment.has_platform(platform_name):
        pytest.skip(f"Platform '{platform_name}' is not enabled.")  # TODO: not enabled -> not installed
    if not user_context.environment.has_target(target_name):
        pytest.skip(f"Target '{target_name}' is not enabled.")  # TODO: not enabled -> not installed
    for feature in feature_names:
        if not user_context.environment.has_feature(feature):
            pytest.skip(f"Feature '{feature}' is not enabled.")
    user_context.environment.paths["models"] = [PathConfig(models_dir)]
    session = user_context.create_session()
    user_config = user_context.environment.vars.copy()
    user_config.update(config)
    run = session.create_run(config=user_config)
    run.add_features_by_name(feature_names, context=user_context)
    run.add_frontend_by_name(frontend_name, context=user_context)
    run.add_model_by_name(model_name, context=user_context)
    run.add_backend_by_name(backend_name, context=user_context)
    run.add_platform_by_name(platform_name, context=user_context)
    run.add_target_by_name(target_name, context=user_context)
    success = session.process_runs(until=RunStage.RUN, context=user_context)
    report = session.get_reports()
    df = report.df
    assert success
    assert len(df) == 1
    assert "trace" in df["Features"][0]
    assert "RAM stack" in df.columns
    assert "RAM heap" in df.columns
    # TODO: check for dyn. memory metrics columns


@pytest.mark.slow
@pytest.mark.user_context
@pytest.mark.parametrize(
    "model_name", DEFAULT_TFLITE_MODELS
)  # TODO: add test model for this, also test with wrong data
@pytest.mark.parametrize("frontend_name", DEFAULT_FRONTENDS)  # Validate is frontend feature as well
@pytest.mark.parametrize(
    "backend_name", ["tvmaot"]  # -> add tvm if we have a test model for this
)  # TODO: Single backend would be fine, but it has to be enabled
@pytest.mark.parametrize("target_name", ["host_x86"])
@pytest.mark.parametrize(
    "platform_name", ["mlif"]
)  # If we would rename host_x86 to linux we could also use espidf here?
@pytest.mark.parametrize("feature_names", [["unpacked_api"]])
@pytest.mark.parametrize("config", [{}])
def test_feature_unpacked_api(
    user_context, frontend_name, model_name, backend_name, target_name, platform_name, models_dir, feature_names, config
):
    if not user_context.environment.has_frontend(frontend_name):
        pytest.skip(f"Frontend '{frontend_name}' is not enabled.")
    if not user_context.environment.has_backend(backend_name):
        pytest.skip(f"Backend '{backend_name}' is not enabled.")
    if not user_context.environment.has_platform(platform_name):
        pytest.skip(f"Platform '{platform_name}' is not enabled.")  # TODO: not enabled -> not installed
    if not user_context.environment.has_target(target_name):
        pytest.skip(f"Target '{target_name}' is not enabled.")  # TODO: not enabled -> not installed
    for feature in feature_names:
        if not user_context.environment.has_feature(feature):
            pytest.skip(f"Feature '{feature}' is not enabled.")
    user_context.environment.paths["models"] = [PathConfig(models_dir)]
    session = user_context.create_session()
    user_config = user_context.environment.vars.copy()
    user_config.update(config)
    run = session.create_run(config=user_config)
    run.add_features_by_name(feature_names, context=user_context)
    run.add_frontend_by_name(frontend_name, context=user_context)
    run.add_model_by_name(model_name, context=user_context)
    run.add_backend_by_name(backend_name, context=user_context)
    # run.add_platform_by_name(platform_name, context=user_context)
    # run.add_target_by_name(target_name, context=user_context)
    success = session.process_runs(until=RunStage.BUILD, context=user_context)
    report = session.get_reports()
    df = report.df
    assert success
    assert len(df) == 1
    assert "unpacked_api" in df["Features"][0]
    # TODO: check generated code -> do not run at all (would need to check for metrics changes)


@pytest.mark.slow
@pytest.mark.user_context
@pytest.mark.parametrize(
    "model_name", DEFAULT_TFLITE_MODELS
)  # TODO: add test model for this, also test with wrong data
@pytest.mark.parametrize("frontend_name", DEFAULT_FRONTENDS)  # Validate is frontend feature as well
@pytest.mark.parametrize(
    "backend_name", ["tvmaot"]  # -> add tvm if we have a test model for this
)  # TODO: Single backend would be fine, but it has to be enabled
@pytest.mark.parametrize("target_name", ["host_x86"])
@pytest.mark.parametrize(
    "platform_name", ["mlif"]
)  # If we would rename host_x86 to linux we could also use espidf here?
@pytest.mark.parametrize("feature_names", [["usmp"]])
@pytest.mark.parametrize(
    "config",
    [{"usmp.algorithm": "greedy_by_size"}, {"usmp.algorithm": "greedy_by_conflicts"}, {"usmp.algorithm": "hill_climb"}],
)
def test_feature_usmp(
    user_context, frontend_name, model_name, backend_name, target_name, platform_name, models_dir, feature_names, config
):
    if not user_context.environment.has_frontend(frontend_name):
        pytest.skip(f"Frontend '{frontend_name}' is not enabled.")
    if not user_context.environment.has_backend(backend_name):
        pytest.skip(f"Backend '{backend_name}' is not enabled.")
    if not user_context.environment.has_platform(platform_name):
        pytest.skip(f"Platform '{platform_name}' is not enabled.")  # TODO: not enabled -> not installed
    if not user_context.environment.has_target(target_name):
        pytest.skip(f"Target '{target_name}' is not enabled.")  # TODO: not enabled -> not installed
    for feature in feature_names:
        if not user_context.environment.has_feature(feature):
            pytest.skip(f"Feature '{feature}' is not enabled.")
    user_context.environment.paths["models"] = [PathConfig(models_dir)]
    session = user_context.create_session()
    user_config = user_context.environment.vars.copy()
    user_config.update(config)
    run = session.create_run(config=user_config)
    run.add_features_by_name(feature_names, context=user_context)
    run.add_frontend_by_name(frontend_name, context=user_context)
    run.add_model_by_name(model_name, context=user_context)
    run.add_backend_by_name(backend_name, context=user_context)
    # run.add_platform_by_name(platform_name, context=user_context)
    # run.add_target_by_name(target_name, context=user_context)
    success = session.process_runs(until=RunStage.BUILD, context=user_context)
    report = session.get_reports()
    df = report.df
    assert success
    assert len(df) == 1
    assert "usmp" in df["Features"][0]
    # TODO: run twice and compare generted code or look for specific code


@pytest.mark.slow
@pytest.mark.user_context
@pytest.mark.parametrize(
    "model_name", DEFAULT_TFLITE_MODELS
)  # TODO: add test model for this, also test with wrong data
@pytest.mark.parametrize("frontend_name", DEFAULT_FRONTENDS)  # Validate is frontend feature as well
@pytest.mark.parametrize(
    "backend_name", ["tvmaot"]  # other tvm backends?
)  # TODO: Single backend would be fine, but it has to be enabled
@pytest.mark.parametrize("target_name", ["host_x86"])
@pytest.mark.parametrize(
    "platform_name", ["mlif"]
)  # If we would rename host_x86 to linux we could also use espidf here?
@pytest.mark.parametrize("feature_names", [["disable_legalize"]])
@pytest.mark.parametrize("config", [{}])
def test_feature_disable_legalize(
    user_context, frontend_name, model_name, backend_name, target_name, platform_name, models_dir, feature_names, config
):
    if not user_context.environment.has_frontend(frontend_name):
        pytest.skip(f"Frontend '{frontend_name}' is not enabled.")
    if not user_context.environment.has_backend(backend_name):
        pytest.skip(f"Backend '{backend_name}' is not enabled.")
    if not user_context.environment.has_platform(platform_name):
        pytest.skip(f"Platform '{platform_name}' is not enabled.")  # TODO: not enabled -> not installed
    if not user_context.environment.has_target(target_name):
        pytest.skip(f"Target '{target_name}' is not enabled.")  # TODO: not enabled -> not installed
    for feature in feature_names:
        if not user_context.environment.has_feature(feature):
            pytest.skip(f"Feature '{feature}' is not enabled.")
    user_context.environment.paths["models"] = [PathConfig(models_dir)]
    session = user_context.create_session()
    user_config = user_context.environment.vars.copy()
    user_config.update(config)
    run = session.create_run(config=user_config)
    run.add_features_by_name(feature_names, context=user_context)
    run.add_frontend_by_name(frontend_name, context=user_context)
    run.add_model_by_name(model_name, context=user_context)
    run.add_backend_by_name(backend_name, context=user_context)
    # run.add_platform_by_name(platform_name, context=user_context)
    # run.add_target_by_name(target_name, context=user_context)
    success = session.process_runs(until=RunStage.BUILD, context=user_context)
    report = session.get_reports()
    df = report.df
    assert success
    assert len(df) == 1
    assert "disable_legalize" in df["Features"][0]
    # TODO: run twice and compare codegen results


@pytest.mark.slow
@pytest.mark.user_context
@pytest.mark.parametrize("model_name", ["sine_model"])  # TODO: add test model for this, also test with wrong data
@pytest.mark.parametrize("frontend_name", ["tflite"])  # Validate is frontend feature as well
@pytest.mark.parametrize(
    "backend_name", ["tvmllvm"]  # other tvm backends?
)  # TODO: Single backend would be fine, but it has to be enabled
@pytest.mark.parametrize("target_name", ["tvm_cpu"])
@pytest.mark.parametrize("feature_names", [["autotune"]])
@pytest.mark.parametrize("config", [{}])
def test_feature_autotune(
    user_context, frontend_name, model_name, backend_name, target_name, models_dir, feature_names, config
):
    platform_name = "tvm"
    if not user_context.environment.has_frontend(frontend_name):
        pytest.skip(f"Frontend '{frontend_name}' is not enabled.")
    if not user_context.environment.has_backend(backend_name):
        pytest.skip(f"Backend '{backend_name}' is not enabled.")
    if not user_context.environment.has_platform(platform_name):
        pytest.skip(f"Platform '{platform_name}' is not enabled.")  # TODO: not enabled -> not installed
    if not user_context.environment.has_target(target_name):
        pytest.skip(f"Target '{target_name}' is not enabled.")  # TODO: remove check?
    for feature in feature_names:
        if not user_context.environment.has_feature(feature):
            pytest.skip(f"Feature '{feature}' is not enabled.")
    user_context.environment.paths["models"] = [PathConfig(models_dir)]
    session = user_context.create_session()
    user_config = user_context.environment.vars.copy()
    user_config.update(config)
    run = session.create_run(config=user_config)
    run.add_features_by_name(feature_names, context=user_context)
    run.add_frontend_by_name(frontend_name, context=user_context)
    run.add_model_by_name(model_name, context=user_context)
    run.add_platform_by_name(platform_name, context=user_context)
    run.add_target_by_name(target_name, context=user_context)
    run.add_backend_by_name(backend_name, context=user_context)
    success = session.process_runs(until=RunStage.TUNE, context=user_context)
    report = session.get_reports()
    df = report.df
    assert success
    assert len(df) == 1
    assert "autotune" in df["Features"][0]


@pytest.mark.slow
@pytest.mark.user_context
@pytest.mark.parametrize("model_name", ["sine_model"])  # TODO: add test model for this, also test with wrong data
@pytest.mark.parametrize("frontend_name", ["tflite"])  # Validate is frontend feature as well
@pytest.mark.parametrize(
    "backend_name", ["tvmllvm"]  # other tvm backends?
)  # TODO: Single backend would be fine, but it has to be enabled
@pytest.mark.parametrize("target_name", ["tvm_cpu"])
@pytest.mark.parametrize("feature_names", [["autotune", "autotuned"]])  # TODO: provide tuning records instead
@pytest.mark.parametrize("config", [{"tvmaot.print_outputs": True}])
def test_feature_autotuned(
    user_context, frontend_name, model_name, backend_name, target_name, models_dir, feature_names, config, tmp_path
):
    platform_name = "tvm"
    if not user_context.environment.has_frontend(frontend_name):
        pytest.skip(f"Frontend '{frontend_name}' is not enabled.")
    if not user_context.environment.has_backend(backend_name):
        pytest.skip(f"Backend '{backend_name}' is not enabled.")
    if not user_context.environment.has_platform(platform_name):
        pytest.skip(f"Platform '{platform_name}' is not enabled.")  # TODO: not enabled -> not installed
    if not user_context.environment.has_target(target_name):
        pytest.skip(f"Target '{target_name}' is not enabled.")  # TODO: remove check?
    for feature in feature_names:
        if not user_context.environment.has_feature(feature):
            pytest.skip(f"Feature '{feature}' is not enabled.")
    user_context.environment.paths["models"] = [PathConfig(models_dir)]
    session = user_context.create_session()
    results_file = tmp_path / "tuning.log"
    results_file.touch()
    config.update({"autotuned.results_file": results_file})
    user_config = user_context.environment.vars.copy()
    user_config.update(config)
    run = session.create_run(config=user_config)
    run.add_features_by_name(feature_names, context=user_context)
    run.add_frontend_by_name(frontend_name, context=user_context)
    run.add_platform_by_name(platform_name, context=user_context)
    run.add_target_by_name(target_name, context=user_context)
    run.add_model_by_name(model_name, context=user_context)
    run.add_backend_by_name(backend_name, context=user_context)
    success = session.process_runs(until=RunStage.BUILD, context=user_context)
    report = session.get_reports()
    df = report.df
    assert success
    assert len(df) == 1
    assert "autotuned" in df["Features"][0]


# TODO:
# cmsisnn -> currently broken
# gdbserver -> hard to test
