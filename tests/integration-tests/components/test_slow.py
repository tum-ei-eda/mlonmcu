from mlonmcu.environment.config import PathConfig
from mlonmcu.session.run import RunStage

import pytest


# TODO: add user_session fixture which handles cleanup via session.discard()


# Frontends

DEFAULT_MODELS = [
    "sine_model",
]  # TODO: make sure that we use quant/float models and several different operators
DEFAULT_FRONTENDS = ["tflite"]  # TODO: needs to match with the DEFAULT_MODELS
DEFAULT_BACKENDS = ["tflmi", "tvmaot"]  # TODO: how about tvmrt/tflmc
DEFAULT_PLATFORMS = ["mlif", "espidf"]
# DEFAULT_MLIF_TARGETS = ["host_x86", "etiss_pulpino", "spike", "ovpsim", "corstone300"]
DEFAULT_MLIF_TARGETS = ["host_x86", "etiss_pulpino", "spike", "corstone300"]
DEFAULT_ESPIDF_TARGETS = ["esp32", "esp32c3"]
DEFAULT_TARGETS = DEFAULT_MLIF_TARGETS + DEFAULT_ESPIDF_TARGETS
# VEXT_TARGETS = ["spike", "ovpsim"]
# RISCV_TARGETS = ["spike", "etiss_pulpino", "ovpsim"]
RISCV_TARGETS = ["spike", "etiss_pulpino"]
VEXT_TARGETS = ["spike"]
PEXT_TARGETS = ["spike"]
ARM_MVEI_TARGETS = ["corstone300"]
ARM_DSP_TARGETS = ["corstone300"]
DEBUG_ARENA_BACKENDS = ["tflmi", "tvmaot", "tvmrt"]

# TVM_EXAMPLE_CONFIG_COMMON = {}
TVM_EXAMPLE_CONFIG_COMMON = {
    "extra_pass_config": {"relay.FuseOps.max_depth": 0},  # TODO
    "disabled_passes": ["AlterOpLayout"],
    "target_device": "arm_cpu",
    "opt_level": 2,
}


@pytest.mark.user_context
@pytest.mark.parametrize("model_name", DEFAULT_MODELS)
@pytest.mark.parametrize("feature_names", [[]])
@pytest.mark.parametrize("config", [{}])
def test_frontend_tflite(user_context, model_name, models_dir, feature_names, config):
    frontend_name = "tflite"
    if not user_context.environment.has_frontend(frontend_name):
        pytest.skip(f"Frontend '{frontend_name}' is not enabled.")
    for feature in feature_names:
        if not user_context.environment.has_feature(feature):
            pytest.skip(f"Feature '{feature}' is not enabled.")
    user_context.environment.paths["models"] = [PathConfig(models_dir)]
    session = user_context.create_session()
    run = session.create_run(config=config)
    run.add_features_by_name(feature_names, context=user_context)
    run.add_frontend_by_name(frontend_name, context=user_context)
    run.add_model_by_name(model_name, context=user_context)
    session.process_runs(until=RunStage.LOAD, context=user_context)
    report = session.get_reports()
    df = report.df
    assert len(df) == 1
    assert df["Model"][0] == model_name
    assert df["Frontend"][0] == frontend_name
    # TODO: check artifacts


# Backends

# TODO: decide if execute on a per-framework basis?


@pytest.mark.user_context
@pytest.mark.parametrize("model_name", DEFAULT_MODELS)
@pytest.mark.parametrize("frontend_name", ["tflite"])
@pytest.mark.parametrize("feature_names", [[]])
@pytest.mark.parametrize(
    "config", [{}, {"arena_size": 2**20, "operators": ["TODO"]}]  # TODO
)  # TODO: user should be ablte to overwrite sesstings parsed by frontend
def test_backend_tflmi(user_context, frontend_name, model_name, models_dir, feature_names, config):
    if not user_context.environment.has_frontend(frontend_name):
        pytest.skip(f"Frontend '{frontend_name}' is not enabled.")
    backend_name = "tflmi"
    if not user_context.environment.has_backend(backend_name):
        pytest.skip(f"Backend '{backend_name}' is not enabled.")
    for feature in feature_names:
        if not user_context.environment.has_feature(feature):
            pytest.skip(f"Feature '{feature}' is not enabled.")
    user_context.environment.paths["models"] = [PathConfig(models_dir)]
    session = user_context.create_session()
    run = session.create_run(config=config)
    run.add_features_by_name(feature_names, context=user_context)
    run.add_frontend_by_name(frontend_name, context=user_context)
    run.add_model_by_name(model_name, context=user_context)
    run.add_backend_by_name(backend_name, context=user_context)  # TODO: implicit Framework
    success = session.process_runs(until=RunStage.BUILD, context=user_context)
    report = session.get_reports()
    df = report.df
    assert success
    assert len(df) == 1
    assert df["Framework"][0] == "tflm"  # TODO: rename to tflm
    assert df["Backend"][0] == backend_name
    # TODO: check artifacts


@pytest.mark.skip("Currently not supported")
@pytest.mark.user_context
@pytest.mark.parametrize("model_name", DEFAULT_MODELS)
@pytest.mark.parametrize("frontend_name", ["tflite"])
@pytest.mark.parametrize("feature_names", [[]])
@pytest.mark.parametrize("config", [{}])
def test_backend_tflmc(user_context, frontend_name, model_name, models_dir, feature_names, config):
    if not user_context.environment.has_frontend(frontend_name):
        pytest.skip(f"Frontend '{frontend_name}' is not enabled.")
    backend_name = "tflmc"
    if not user_context.environment.has_backend(backend_name):
        pytest.skip(f"Backend '{backend_name}' is not enabled.")
    for feature in feature_names:
        if not user_context.environment.has_feature(feature):
            pytest.skip(f"Feature '{feature}' is not enabled.")
    user_context.environment.paths["models"] = [PathConfig(models_dir)]
    session = user_context.create_session()
    run = session.create_run(config=config)
    run.add_features_by_name(feature_names, context=user_context)
    run.add_frontend_by_name(frontend_name, context=user_context)
    run.add_model_by_name(model_name, context=user_context)
    run.add_backend_by_name(backend_name, context=user_context)  # TODO: implicit Framework
    success = session.process_runs(until=RunStage.BUILD, context=user_context)
    report = session.get_reports()
    df = report.df
    assert success
    assert len(df) == 1
    assert df["Framework"][0] == "tflm"  # TODO: rename to tflm
    assert df["Backend"][0] == backend_name
    # TODO: check artifacts


@pytest.mark.slow
@pytest.mark.context
@pytest.mark.parametrize("model_name", DEFAULT_MODELS)
@pytest.mark.parametrize("frontend_name", ["tflite"])
@pytest.mark.parametrize("feature_names", [[]])
@pytest.mark.parametrize(
    "config",
    [
        {},
        {
            **TVM_EXAMPLE_CONFIG_COMMON,
            "arena_size": 2**20,
            "alignment_bytes": 16,
        },
    ],
)
def test_backend_tvmaot(user_context, frontend_name, model_name, models_dir, feature_names, config):
    if not user_context.environment.has_frontend(frontend_name):
        pytest.skip(f"Frontend '{frontend_name}' is not enabled.")
    backend_name = "tvmaot"
    if not user_context.environment.has_backend(backend_name):
        pytest.skip(f"Backend '{backend_name}' is not enabled.")
    for feature in feature_names:
        if not user_context.environment.has_feature(feature):
            pytest.skip(f"Feature '{feature}' is not enabled.")
    user_context.environment.paths["models"] = [PathConfig(models_dir)]
    config = {f"{backend_name}.{key}": value for key, value in config.items()}
    session = user_context.create_session()
    run = session.create_run(config=config)
    run.add_features_by_name(feature_names, context=user_context)
    run.add_frontend_by_name(frontend_name, context=user_context)
    run.add_model_by_name(model_name, context=user_context)
    run.add_backend_by_name(backend_name, context=user_context)
    success = session.process_runs(until=RunStage.BUILD, context=user_context)
    report = session.get_reports()
    df = report.df
    assert success
    assert len(df) == 1
    assert df["Framework"][0] == "tvm"
    assert df["Backend"][0] == backend_name
    # TODO: check artifacts


@pytest.mark.slow
@pytest.mark.user_context
@pytest.mark.parametrize("model_name", DEFAULT_MODELS)
@pytest.mark.parametrize("frontend_name", ["tflite"])
@pytest.mark.parametrize("feature_names", [[]])
@pytest.mark.parametrize(
    "config",
    [
        {},
        {
            **TVM_EXAMPLE_CONFIG_COMMON,
            "arena_size": 2**20,
        },
    ],
)  # TODO: combine tvm common configs
def test_backend_tvmrt(user_context, frontend_name, model_name, models_dir, feature_names, config):
    if not user_context.environment.has_frontend(frontend_name):
        pytest.skip(f"Frontend '{frontend_name}' is not enabled.")
    backend_name = "tvmrt"
    if not user_context.environment.has_backend(backend_name):
        pytest.skip(f"Backend '{backend_name}' is not enabled.")
    user_context.environment.paths["models"] = [PathConfig(models_dir)]
    config = {f"{backend_name}.{key}": value for key, value in config.items()}
    session = user_context.create_session()
    for feature in feature_names:
        if not user_context.environment.has_feature(feature):
            pytest.skip(f"Feature '{feature}' is not enabled.")
    run = session.create_run(config=config)
    run.add_features_by_name(feature_names, context=user_context)
    run.add_frontend_by_name(frontend_name, context=user_context)
    run.add_model_by_name(model_name, context=user_context)
    run.add_backend_by_name(backend_name, context=user_context)
    success = session.process_runs(until=RunStage.BUILD, context=user_context)
    report = session.get_reports()
    df = report.df
    assert success
    assert len(df) == 1
    assert df["Framework"][0] == "tvm"
    assert df["Backend"][0] == backend_name
    # TODO: check artifacts
    # TODO: check arena and operators


@pytest.mark.slow
@pytest.mark.user_context
@pytest.mark.parametrize("model_name", DEFAULT_MODELS)
@pytest.mark.parametrize("frontend_name", ["tflite"])
@pytest.mark.parametrize("feature_names", [[]])
@pytest.mark.parametrize(
    "config",
    [
        {},
        {
            **TVM_EXAMPLE_CONFIG_COMMON,
            "arena_size": 2**20,
        },
    ],
)
def test_backend_tvmcg(user_context, frontend_name, model_name, models_dir, feature_names, config):
    if not user_context.environment.has_frontend(frontend_name):
        pytest.skip(f"Frontend '{frontend_name}' is not enabled.")
    backend_name = "tvmcg"
    if not user_context.environment.has_backend(backend_name):
        pytest.skip(f"Backend '{backend_name}' is not enabled.")
    for feature in feature_names:
        if not user_context.environment.has_feature(feature):
            pytest.skip(f"Feature '{feature}' is not enabled.")
    user_context.environment.paths["models"] = [PathConfig(models_dir)]
    config = {f"{backend_name}.{key}": value for key, value in config.items()}
    session = user_context.create_session()
    run = session.create_run(config=config)
    run.add_features_by_name(feature_names, context=user_context)
    run.add_frontend_by_name(frontend_name, context=user_context)
    run.add_model_by_name(model_name, context=user_context)
    run.add_backend_by_name(backend_name, context=user_context)
    success = session.process_runs(until=RunStage.BUILD, context=user_context)
    report = session.get_reports()
    df = report.df
    assert success
    assert len(df) == 1
    assert df["Framework"][0] == "tvm"
    assert df["Backend"][0] == backend_name
    # TODO: check artifacts


# Platforms(Compile)/Targets(Run)
@pytest.mark.slow
@pytest.mark.user_context
@pytest.mark.parametrize("model_name", DEFAULT_MODELS)
@pytest.mark.parametrize("frontend_name", DEFAULT_FRONTENDS)
@pytest.mark.parametrize("backend_name", DEFAULT_BACKENDS)
@pytest.mark.parametrize("target_name", DEFAULT_MLIF_TARGETS)
@pytest.mark.parametrize("feature_names", [[]])
@pytest.mark.parametrize(
    "config", [{"tflmi.arena_size": 2**17, "tvmaot.arena_size": 2**17}]
)  # corstone300 has limited RAM, TODO: find a better way!
def test_platform_mlif(
    user_context, frontend_name, model_name, backend_name, target_name, models_dir, feature_names, config
):
    if not user_context.environment.has_frontend(frontend_name):
        pytest.skip(f"Frontend '{frontend_name}' is not enabled.")
    if not user_context.environment.has_backend(backend_name):
        pytest.skip(f"Backend '{backend_name}' is not enabled.")
    platform_name = "mlif"
    if not user_context.environment.has_platform(platform_name):
        pytest.skip(f"Platform '{platform_name}' is not enabled.")  # TODO: not enabled -> not installed
    if not user_context.environment.has_target(target_name):
        pytest.skip(f"Target '{target_name}' is not enabled.")  # TODO: remove check?
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
    success = session.process_runs(until=RunStage.COMPILE, context=user_context)
    report = session.get_reports()
    df = report.df
    assert success
    assert len(df) == 1
    assert df["Platform"][0] == platform_name
    assert df["Target"][0] == target_name
    # TODO: check artifacts


@pytest.mark.slow
@pytest.mark.user_context
@pytest.mark.parametrize("model_name", DEFAULT_MODELS)
@pytest.mark.parametrize("frontend_name", DEFAULT_FRONTENDS)
@pytest.mark.parametrize("backend_name", DEFAULT_BACKENDS)
@pytest.mark.parametrize("target_name", DEFAULT_MLIF_TARGETS)
@pytest.mark.parametrize("feature_names", [[]])
@pytest.mark.parametrize(
    "config", [{"tflmi.arena_size": 2**17, "tvmaot.arena_size": 2**17}]
)  # corstone300 has limited RAM, TODO: find a better way!
def test_target_mlif(
    user_context, frontend_name, model_name, backend_name, target_name, models_dir, feature_names, config
):
    if not user_context.environment.has_frontend(frontend_name):
        pytest.skip(f"Frontend '{frontend_name}' is not enabled.")
    if not user_context.environment.has_backend(backend_name):
        pytest.skip(f"Backend '{backend_name}' is not enabled.")
    platform_name = "mlif"
    if not user_context.environment.has_platform(platform_name):
        pytest.skip(f"Platform '{platform_name}' is not enabled.")  # TODO: not enabled -> not installed
    if not user_context.environment.has_target(target_name):
        pytest.skip(f"Target '{target_name}' is not enabled.")  # TODO: remove check?
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
    assert df["Platform"][0] == platform_name
    assert df["Target"][0] == target_name
    # TODO: check artifacts


# TODO: etiss_verbose!


@pytest.mark.slow
@pytest.mark.user_context
@pytest.mark.parametrize("model_name", DEFAULT_MODELS)
@pytest.mark.parametrize("frontend_name", DEFAULT_FRONTENDS)
# @pytest.mark.parametrize("backend_name", DEFAULT_BACKENDS)
@pytest.mark.parametrize("backend_name", ["tflmi", "tvmaot"])  # TODO: fix tvmrt support in ESP-IDF
@pytest.mark.parametrize("target_name", DEFAULT_ESPIDF_TARGETS)
@pytest.mark.parametrize("feature_names", [[]])
@pytest.mark.parametrize(
    "config", [{"espidf.wait_for_user": False, "tflmi.arena_size": 2**17, "tvmaot.arena_size": 2**17}]
)
def test_platform_espidf(
    user_context, frontend_name, model_name, backend_name, target_name, models_dir, feature_names, config
):
    if not user_context.environment.has_frontend(frontend_name):
        pytest.skip(f"Frontend '{frontend_name}' is not enabled.")
    if not user_context.environment.has_backend(backend_name):
        pytest.skip(f"Backend '{backend_name}' is not enabled.")
    platform_name = "espidf"
    if not user_context.environment.has_platform(platform_name):
        pytest.skip(f"Platform '{platform_name}' is not enabled.")  # TODO: not enabled -> not installed
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
    success = session.process_runs(until=RunStage.COMPILE, context=user_context)
    report = session.get_reports()
    df = report.df
    assert success
    assert len(df) == 1
    assert df["Platform"][0] == platform_name
    assert df["Target"][0] == target_name
    # TODO: check artifacts


@pytest.mark.slow
@pytest.mark.hardware
@pytest.mark.user_context
@pytest.mark.parametrize("model_name", DEFAULT_MODELS)
@pytest.mark.parametrize("frontend_name", DEFAULT_FRONTENDS)
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
def test_target_espidf(
    user_context, frontend_name, model_name, backend_name, target_name, models_dir, feature_names, config
):
    if not user_context.environment.has_frontend(frontend_name):
        pytest.skip(f"Frontend '{frontend_name}' is not enabled.")
    if not user_context.environment.has_backend(backend_name):
        pytest.skip(f"Backend '{backend_name}' is not enabled.")
    platform_name = "espidf"
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
    assert df["Platform"][0] == platform_name
    assert df["Target"][0] == target_name
    # TODO: check artifacts


@pytest.mark.slow
@pytest.mark.hardware
@pytest.mark.user_context
@pytest.mark.parametrize("model_name", DEFAULT_MODELS)
@pytest.mark.parametrize("frontend_name", DEFAULT_FRONTENDS)
@pytest.mark.parametrize("backend_name", ["tvmllvm"])
@pytest.mark.parametrize("target_name", ["tvm_cpu"])
@pytest.mark.parametrize("feature_names", [[]])
@pytest.mark.parametrize(
    "config",
    [{}],
)
def test_target_tvm(
    user_context, frontend_name, model_name, backend_name, target_name, models_dir, feature_names, config
):
    if not user_context.environment.has_frontend(frontend_name):
        pytest.skip(f"Frontend '{frontend_name}' is not enabled.")
    if not user_context.environment.has_backend(backend_name):
        pytest.skip(f"Backend '{backend_name}' is not enabled.")
    platform_name = "espidf"
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
    assert df["Platform"][0] == platform_name
    assert df["Target"][0] == target_name
    # TODO: check artifacts


# TODO: test microtvm platforn and targets - Needs zephyr or arduino installed -> Manually or via mlonmcu?

# # PostProcesses


# Features
@pytest.mark.slow
@pytest.mark.user_context
@pytest.mark.parametrize("model_name", DEFAULT_MODELS)  # TODO: single model would be enough
@pytest.mark.parametrize("frontend_name", DEFAULT_FRONTENDS)
@pytest.mark.parametrize("backend_name", DEBUG_ARENA_BACKENDS)
@pytest.mark.parametrize("target_name", ["host_x86"])
@pytest.mark.parametrize("platform_name", ["mlif"])
@pytest.mark.parametrize(
    "feature_names", [["debug_arena", "debug"]]
)  # TODO: should debug_arena set {target}.print_outputs automatically?
@pytest.mark.parametrize(
    "config", [{"host_x86.print_outputs": True}]
)  # TODO: get rid of this by writing stdout to an artifact/file
def test_feature_debug_arena(
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
    assert "debug_arena" in df["Features"][0]
    # TODO: check artifacts
    # Check generated code
    # Check stdout


@pytest.mark.slow
@pytest.mark.user_context
@pytest.mark.parametrize("model_name", DEFAULT_MODELS)  # TODO: add test model for this, also test with wrong data
@pytest.mark.parametrize("frontend_name", DEFAULT_FRONTENDS)  # Validate is frontend feature as well
@pytest.mark.parametrize(
    "backend_name", ["tflmi"]  # -> add tvm if we have a test model for this
)  # TODO: Single backend would be fine, but it has to be enabled
@pytest.mark.parametrize("target_name", ["etiss_pulpino"])
@pytest.mark.parametrize("platform_name", ["mlif"])  # TODO: add validate to espidf and test this as well
@pytest.mark.parametrize("feature_names", [["validate", "debug"]])  # currently validate does not imply debug
@pytest.mark.parametrize(
    "config", [{"host_x86.print_outputs": True}]  # We do not ned this if we just use the report col
)
def test_feature_validate(
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
    assert "validate" in df["Features"][0]
    assert "Validation" in df.columns
    assert df["Validation"][0]  # if model has validation data else, missing/NaN/None?
    # TODO: force correct
    # TODO: force missmatch
    # TODO: force missing
    # TODO: force invalid size


@pytest.mark.slow
@pytest.mark.user_context
@pytest.mark.parametrize("model_name", DEFAULT_MODELS)  # TODO: add test model for this, also test with wrong data
@pytest.mark.parametrize("frontend_name", DEFAULT_FRONTENDS)  # Validate is frontend feature as well
@pytest.mark.parametrize(
    "backend_name", ["tflmi"]  # -> add tvm if we have a test model for this
)  # TODO: Single backend would be fine, but it has to be enabled
@pytest.mark.parametrize("target_name", ["host_x86"])
@pytest.mark.parametrize(
    "platform_name", ["mlif"]
)  # If we would rename host_x86 to linux we could also use espidf here?
@pytest.mark.parametrize("feature_names", [["debug"]])  # currently validate does not imply debug
@pytest.mark.parametrize(
    "config", [{"host_x86.print_outputs": True}]  # We do not ned this if we just use the report col
)
def test_feature_debug(
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
    assert "debug" in df["Features"][0]
    # TODO: stdout with test model
    # TODO: 2 runs to compare ROM/RAM/Cycles?


@pytest.mark.slow
@pytest.mark.user_context
@pytest.mark.parametrize("model_name", DEFAULT_MODELS)  # TODO: add test model for this, also test with wrong data
@pytest.mark.parametrize("frontend_name", DEFAULT_FRONTENDS)  # Validate is frontend feature as well
@pytest.mark.parametrize(
    "backend_name", ["tflmi"]  # -> add tvm if we have a test model for this
)  # TODO: Single backend would be fine, but it has to be enabled
@pytest.mark.parametrize("target_name", DEFAULT_MLIF_TARGETS)
@pytest.mark.parametrize(
    "platform_name", ["mlif"]
)  # If we would rename host_x86 to linux we could also use espidf here?
@pytest.mark.parametrize(
    "feature_names", [["muriscvnn"], ["muriscvnn", "debug"]]
)  # currently validate does not imply debug
@pytest.mark.parametrize("config", [{}])
def test_feature_muriscvnn(
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
    assert "muriscvnn" in df["Features"][0]
    # TODO: find out if kernels are actually linked?
    # TODO: 2 runs to compare ROM/RAM/Cycles?


@pytest.mark.slow
@pytest.mark.user_context
@pytest.mark.parametrize("model_name", DEFAULT_MODELS)  # TODO: add test model for this, also test with wrong data
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
@pytest.mark.parametrize("model_name", DEFAULT_MODELS)  # TODO: add test model for this, also test with wrong data
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
@pytest.mark.parametrize("model_name", DEFAULT_MODELS)  # TODO: add test model for this, also test with wrong data
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
@pytest.mark.parametrize("model_name", DEFAULT_MODELS)  # TODO: add test model for this, also test with wrong data
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
    assert "vext" in df["Features"][0]
    # TODO: find out if kernels are actually linked?
    # TODO: 2 runs to compare ROM/RAM/Cycles?


@pytest.mark.slow
@pytest.mark.user_context
@pytest.mark.parametrize("model_name", DEFAULT_MODELS)  # TODO: add test model for this, also test with wrong data
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
    assert "pext" in df["Features"][0]
    # TODO: find out if kernels are actually linked?
    # TODO: 2 runs to compare ROM/RAM/Cycles?


@pytest.mark.slow
@pytest.mark.user_context
@pytest.mark.parametrize("model_name", DEFAULT_MODELS)  # TODO: add test model for this, also test with wrong data
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
    assert "arm_mvei" in df["Features"][0]
    # TODO: find out if kernels are actually linked?
    # TODO: 2 runs to compare ROM/RAM/Cycles?


@pytest.mark.slow
@pytest.mark.user_context
@pytest.mark.parametrize("model_name", DEFAULT_MODELS)  # TODO: add test model for this, also test with wrong data
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
    assert "arm_dsp" in df["Features"][0]
    # TODO: find out if kernels are actually linked?
    # TODO: 2 runs to compare ROM/RAM/Cycles?


@pytest.mark.slow
@pytest.mark.user_context
@pytest.mark.parametrize("model_name", DEFAULT_MODELS)  # TODO: add test model for this, also test with wrong data
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
@pytest.mark.parametrize("model_name", DEFAULT_MODELS)  # TODO: add test model for this, also test with wrong data
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
    assert "trace" in df["Features"][0]
    assert "RAM stack" in df.columns
    assert "RAM heap" in df.columns
    # TODO: check for dyn. memory metrics columns


@pytest.mark.slow
@pytest.mark.user_context
@pytest.mark.parametrize("model_name", DEFAULT_MODELS)  # TODO: add test model for this, also test with wrong data
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
    run = session.create_run(config=config)
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
@pytest.mark.parametrize("model_name", DEFAULT_MODELS)  # TODO: add test model for this, also test with wrong data
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
    run = session.create_run(config=config)
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
@pytest.mark.parametrize("model_name", DEFAULT_MODELS)  # TODO: add test model for this, also test with wrong data
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
    run = session.create_run(config=config)
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
    "backend_name", ["tvmaot"]  # other tvm backends?
)  # TODO: Single backend would be fine, but it has to be enabled
@pytest.mark.parametrize("feature_names", [["autotune"]])
@pytest.mark.parametrize("config", [{}])
def test_feature_autotune(user_context, frontend_name, model_name, backend_name, models_dir, feature_names, config):
    if not user_context.environment.has_frontend(frontend_name):
        pytest.skip(f"Frontend '{frontend_name}' is not enabled.")
    if not user_context.environment.has_backend(backend_name):
        pytest.skip(f"Backend '{backend_name}' is not enabled.")
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
    "backend_name", ["tvmaot"]  # other tvm backends?
)  # TODO: Single backend would be fine, but it has to be enabled
@pytest.mark.parametrize("feature_names", [["autotune", "autotuned"]])  # TODO: provide tuning records instead
@pytest.mark.parametrize("config", [{"tvmaot.print_outputs": True}])
def test_feature_autotuned(
    user_context, frontend_name, model_name, backend_name, models_dir, feature_names, config, tmp_path
):
    if not user_context.environment.has_frontend(frontend_name):
        pytest.skip(f"Frontend '{frontend_name}' is not enabled.")
    if not user_context.environment.has_backend(backend_name):
        pytest.skip(f"Backend '{backend_name}' is not enabled.")
    for feature in feature_names:
        if not user_context.environment.has_feature(feature):
            pytest.skip(f"Feature '{feature}' is not enabled.")
    user_context.environment.paths["models"] = [PathConfig(models_dir)]
    session = user_context.create_session()
    results_file = tmp_path / "tuning.log"
    results_file.touch()
    config.update({"autotuned.results_file": results_file})
    run = session.create_run(config=config)
    run.add_features_by_name(feature_names, context=user_context)
    run.add_frontend_by_name(frontend_name, context=user_context)
    run.add_model_by_name(model_name, context=user_context)
    run.add_backend_by_name(backend_name, context=user_context)
    success = session.process_runs(until=RunStage.BUILD, context=user_context)
    report = session.get_reports()
    df = report.df
    print("artifacts", run.artifacts_per_stage)
    assert success
    assert len(df) == 1
    assert "autotuned" in df["Features"][0]


# TODO:
# cmsisnn -> currently broken
# gdbserver -> hard to test
