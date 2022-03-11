from time import sleep
from mlonmcu.environment.config import PathConfig
from mlonmcu.session.run import RunStage

import pytest


# def test_func_fast():
#     sleep(0.1)
#
#
# @pytest.mark.slow
# def test_func_slow():
#     sleep(10)


# TODO: add user_session fixture which handles cleanup via session.discard()


# Frontends

DEFAULT_MODELS = ["sine_model"]
DEFAULT_FRONTENDS = ["tflite"]
DEFAULT_BACKENDS = ["tflmi", "tvmaot"]
DEFAULT_PLATFORMS = ["mlif", "espidf"]
# DEFAULT_MLIF_TARGETS = ["host_x86", "etiss_pulpino", "spike", "ovpsim", "corstone300"]
DEFAULT_MLIF_TARGETS = ["host_x86", "etiss_pulpino", "spike", "corstone300"]
DEFAULT_ESPIDF_TARGETS = ["esp32", "esp32c3"]
DEFAULT_TARGETS = DEFAULT_MLIF_TARGETS + DEFAULT_ESPIDF_TARGETS

TVM_EXAMPLE_CONFIG_COMMON = {
    "extra_pass_config": {"relay.FuseOps.max_depth": 0},  # TODO
    "disabled_passes": ["AlterOpLayout"],
    "target_device": "arm_cpu",
    "opt_level": 2,
}


@pytest.mark.user_context
@pytest.mark.parametrize("model_name", DEFAULT_MODELS)
@pytest.mark.parametrize("features", [[]])
@pytest.mark.parametrize("config", [{}])
def test_frontend_tflite(user_context, model_name, models_dir, features, config):
    frontend_name = "tflite"
    if not user_context.environment.has_frontend(frontend_name):
        pytest.skip(f"Frontend '{frontend_name}' is not enabled.")
    user_context.environment.paths["models"] = [PathConfig(models_dir)]
    session = user_context.create_session()
    run = session.create_run(features=features, config=config)
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
@pytest.mark.parametrize("features", [[]])
@pytest.mark.parametrize(
    "config", [{}, {"arena_size": 2 ** 20, "operators": ["TODO"]}]  # TODO
)  # TODO: user should be ablte to overwrite sesstings parsed by frontend
def test_backend_tflmi(user_context, frontend_name, model_name, models_dir, features, config):
    if not user_context.environment.has_frontend(frontend_name):
        pytest.skip(f"Frontend '{frontend_name}' is not enabled.")
    backend_name = "tflmi"
    if not user_context.environment.has_backend(backend_name):
        pytest.skip(f"Backend '{backend_name}' is not enabled.")
    user_context.environment.paths["models"] = [PathConfig(models_dir)]
    session = user_context.create_session()
    run = session.create_run(features=features, config=config)
    run.add_frontend_by_name(frontend_name, context=user_context)
    run.add_model_by_name(model_name, context=user_context)
    run.add_backend_by_name(backend_name, context=user_context)  # TODO: implicit Framework
    success = session.process_runs(until=RunStage.BUILD, context=user_context)
    report = session.get_reports()
    df = report.df
    assert success
    assert len(df) == 1
    assert df["Framework"][0] == "tflite"  # TODO: rename to tflm
    assert df["Backend"][0] == backend_name
    # TODO: check artifacts


@pytest.mark.skip("Currently not supported")
@pytest.mark.user_context
@pytest.mark.parametrize("model_name", DEFAULT_MODELS)
@pytest.mark.parametrize("frontend_name", ["tflite"])
@pytest.mark.parametrize("features", [[]])
@pytest.mark.parametrize("config", [{}])
def test_backend_tflmc(user_context, frontend_name, model_name, models_dir, features, config):
    if not user_context.environment.has_frontend(frontend_name):
        pytest.skip(f"Frontend '{frontend_name}' is not enabled.")
    backend_name = "tflmc"
    if not user_context.environment.has_backend(backend_name):
        pytest.skip(f"Backend '{backend_name}' is not enabled.")
    user_context.environment.paths["models"] = [PathConfig(models_dir)]
    session = user_context.create_session()
    run = session.create_run(features=features, config=config)
    run.add_frontend_by_name(frontend_name, context=user_context)
    run.add_model_by_name(model_name, context=user_context)
    run.add_backend_by_name(backend_name, context=user_context)  # TODO: implicit Framework
    success = session.process_runs(until=RunStage.BUILD, context=user_context)
    report = session.get_reports()
    df = report.df
    assert success
    assert len(df) == 1
    assert df["Framework"][0] == "tflite"  # TODO: rename to tflm
    assert df["Backend"][0] == backend_name
    # TODO: check artifacts


@pytest.mark.slow
@pytest.mark.context
@pytest.mark.parametrize("model_name", DEFAULT_MODELS)
@pytest.mark.parametrize("frontend_name", ["tflite"])
@pytest.mark.parametrize("features", [[]])
@pytest.mark.parametrize(
    "config",
    [
        {},
        {
            **TVM_EXAMPLE_CONFIG_COMMON,
            "arena_size": 2 * 20,
            "alignment_bytes": 16,
        },
    ],
)
def test_backend_tvmaot(user_context, frontend_name, model_name, models_dir, features, config):
    if not user_context.environment.has_frontend(frontend_name):
        pytest.skip(f"Frontend '{frontend_name}' is not enabled.")
    backend_name = "tvmaot"
    if not user_context.environment.has_backend(backend_name):
        pytest.skip(f"Backend '{backend_name}' is not enabled.")
    user_context.environment.paths["models"] = [PathConfig(models_dir)]
    config = {f"{backend_name}.{key}": value for key, value in config.items()}
    session = user_context.create_session()
    run = session.create_run(features=features, config=config)
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
@pytest.mark.parametrize("features", [[]])
@pytest.mark.parametrize(
    "config",
    [
        {},
        {
            **TVM_EXAMPLE_CONFIG_COMMON,
            "arena_size": 2 * 20,
        },
    ],
)  # TODO: combine tvm common configs
def test_backend_tvmrt(user_context, frontend_name, model_name, models_dir, features, config):
    if not user_context.environment.has_frontend(frontend_name):
        pytest.skip(f"Frontend '{frontend_name}' is not enabled.")
    backend_name = "tvmrt"
    if not user_context.environment.has_backend(backend_name):
        pytest.skip(f"Backend '{backend_name}' is not enabled.")
    user_context.environment.paths["models"] = [PathConfig(models_dir)]
    config = {f"{backend_name}.{key}": value for key, value in config.items()}
    session = user_context.create_session()
    run = session.create_run(features=features, config=config)
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
@pytest.mark.parametrize("features", [[]])
@pytest.mark.parametrize(
    "config",
    [
        {},
        {
            **TVM_EXAMPLE_CONFIG_COMMON,
            "arena_size": 2 * 20,
        },
    ],
)
def test_backend_tvmcg(user_context, frontend_name, model_name, models_dir, features, config):
    if not user_context.environment.has_frontend(frontend_name):
        pytest.skip(f"Frontend '{frontend_name}' is not enabled.")
    backend_name = "tvmcg"
    if not user_context.environment.has_backend(backend_name):
        pytest.skip(f"Backend '{backend_name}' is not enabled.")
    user_context.environment.paths["models"] = [PathConfig(models_dir)]
    config = {f"{backend_name}.{key}": value for key, value in config.items()}
    session = user_context.create_session()
    run = session.create_run(features=features, config=config)
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
@pytest.mark.parametrize("features", [[]])
@pytest.mark.parametrize("config", [{}])
def test_platform_mlif(
    user_context, frontend_name, model_name, backend_name, target_name, models_dir, features, config
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
    user_context.environment.paths["models"] = [PathConfig(models_dir)]
    session = user_context.create_session()
    run = session.create_run(features=features, config=config)
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
@pytest.mark.parametrize("features", [[]])
@pytest.mark.parametrize("config", [{}])
def test_target_mlif(user_context, frontend_name, model_name, backend_name, target_name, models_dir, features, config):
    if not user_context.environment.has_frontend(frontend_name):
        pytest.skip(f"Frontend '{frontend_name}' is not enabled.")
    if not user_context.environment.has_backend(backend_name):
        pytest.skip(f"Backend '{backend_name}' is not enabled.")
    platform_name = "mlif"
    if not user_context.environment.has_platform(platform_name):
        pytest.skip(f"Platform '{platform_name}' is not enabled.")  # TODO: not enabled -> not installed
    if not user_context.environment.has_target(target_name):
        pytest.skip(f"Target '{target_name}' is not enabled.")  # TODO: remove check?
    user_context.environment.paths["models"] = [PathConfig(models_dir)]
    session = user_context.create_session()
    run = session.create_run(features=features, config=config)
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
@pytest.mark.user_context
@pytest.mark.parametrize("model_name", DEFAULT_MODELS)
@pytest.mark.parametrize("frontend_name", DEFAULT_FRONTENDS)
@pytest.mark.parametrize("backend_name", DEFAULT_BACKENDS)
@pytest.mark.parametrize("target_name", DEFAULT_ESPIDF_TARGETS)
@pytest.mark.parametrize("features", [[]])
@pytest.mark.parametrize("config", [{"espidf.wait_for_user": False}])
def test_platform_espidf(
    user_context, frontend_name, model_name, backend_name, target_name, models_dir, features, config
):
    if not user_context.environment.has_frontend(frontend_name):
        pytest.skip(f"Frontend '{frontend_name}' is not enabled.")
    if not user_context.environment.has_backend(backend_name):
        pytest.skip(f"Backend '{backend_name}' is not enabled.")
    platform_name = "espidf"
    if not user_context.environment.has_platform(platform_name):
        pytest.skip(f"Platform '{platform_name}' is not enabled.")  # TODO: not enabled -> not installed
    user_context.environment.paths["models"] = [PathConfig(models_dir)]
    session = user_context.create_session()
    run = session.create_run(features=features, config=config)
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


# @pytest.mark.slow
# @pytest.mark.hardware
# @pytest.mark.user_context
# @pytest.mark.parametrize("model_name", DEFAULT_MODELS)
# @pytest.mark.parametrize("frontend_name", DEFAULT_FRONTENDS)
# @pytest.mark.parametrize("backend_name", DEFAULT_BACKENDS)
# @pytest.mark.parametrize("target_name", ["esp32c3"])
# @pytest.mark.parametrize("features", [[]])
# @pytest.mark.parametrize(
#     "config", [{"espidf.wait_for_user": False, "espidf.use_idf_monitor": False, "espidf.port": "/dev/ttyUSB0"}]
# )
# def test_target_espidf(
#     user_context, frontend_name, model_name, backend_name, target_name, models_dir, features, config
# ):
#     if not user_context.environment.has_frontend(frontend_name):
#         pytest.skip(f"Frontend '{frontend_name}' is not enabled.")
#     if not user_context.environment.has_backend(backend_name):
#         pytest.skip(f"Backend '{backend_name}' is not enabled.")
#     platform_name = "espidf"
#     if not user_context.environment.has_platform(platform_name):
#         pytest.skip(f"Platform '{platform_name}' is not enabled.")  # TODO: not enabled -> not installed
#     user_context.environment.paths["models"] = [PathConfig(models_dir)]
#     session = user_context.create_session()
#     run = session.create_run(features=features, config=config)
#     run.add_frontend_by_name(frontend_name, context=user_context)
#     run.add_model_by_name(model_name, context=user_context)
#     run.add_backend_by_name(backend_name, context=user_context)
#     run.add_platform_by_name(platform_name, context=user_context)
#     run.add_target_by_name(target_name, context=user_context)
#     success = session.process_runs(until=RunStage.RUN, context=user_context)
#     report = session.get_reports()
#     df = report.df
#     assert success
#     assert len(df) == 1
#     assert df["Platform"][0] == platform_name
#     assert df["Target"][0] == target_name
#     # TODO: check artifacts
#
#
# # PostProcesses
