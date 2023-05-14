import pytest
from mlonmcu.environment.config import PathConfig

from mlonmcu.session.run import RunStage
from mlonmcu.testing.helpers import DEFAULT_TFLITE_MODELS, DEFAULT_MLIF_TARGETS, _test_run_platform, _test_backend

# Run

# Frontend

# Backend + ?


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
@pytest.mark.parametrize(
    "backend_name", ["tflmi"]  # -> add tvm if we have a test model for this
)  # TODO: Single backend would be fine, but it has to be enabled
@pytest.mark.parametrize("target_name", DEFAULT_MLIF_TARGETS)  # TODO: more targets (without vext)
@pytest.mark.parametrize(
    "feature_names", [["cmsisnn"], ["cmsisnn", "debug"]]
)  # currently validate does not imply debug
@pytest.mark.parametrize("config", [{}])
def test_feature_cmsisnn(user_context, model_name, backend_name, target_name, models_dir, feature_names, config):
    df, artifacts = _test_run_platform(
        None, backend_name, target_name, user_context, model_name, models_dir, feature_names, config
    )
    assert "cmsisnn" in df["Features"][0]
    # TODO: find out if kernels are actually linked?
    # TODO: 2 runs to compare ROM/RAM/Cycles?


@pytest.mark.slow
@pytest.mark.user_context
@pytest.mark.parametrize(
    "model_name", DEFAULT_TFLITE_MODELS
)  # TODO: add test model for this, also test with wrong data
@pytest.mark.parametrize(
    "backend_name", ["tvmaot"]  # -> add tvm if we have a test model for this
)  # TODO: Single backend would be fine, but it has to be enabled
@pytest.mark.parametrize("target_name", DEFAULT_MLIF_TARGETS)
@pytest.mark.parametrize(
    "feature_names", [["muriscvnnbyoc"], ["muriscvnnbyoc", "debug"]]
)  # currently validate does not imply debug
@pytest.mark.parametrize("config", [{}])
def test_feature_muriscvnnbyoc(user_context, model_name, backend_name, target_name, models_dir, feature_names, config):
    df, artifacts = _test_run_platform(
        None, backend_name, target_name, user_context, model_name, models_dir, feature_names, config
    )
    assert "muriscvnnbyoc" in df["Features"][0]
    # TODO: find out if kernels are actually linked?
    # TODO: 2 runs to compare ROM/RAM/Cycles?


@pytest.mark.slow
@pytest.mark.user_context
@pytest.mark.parametrize(
    "model_name", DEFAULT_TFLITE_MODELS
)  # TODO: add test model for this, also test with wrong data
@pytest.mark.parametrize(
    "backend_name", ["tvmaot"]  # -> add tvm if we have a test model for this
)  # TODO: Single backend would be fine, but it has to be enabled
@pytest.mark.parametrize("target_name", DEFAULT_MLIF_TARGETS)  # TODO: more targets (without vext)
@pytest.mark.parametrize(
    "feature_names", [["cmsisnnbyoc"], ["cmsisnnbyoc", "debug"]]
)  # currently validate does not imply debug
@pytest.mark.parametrize("config", [{}])
def test_feature_cmsisnnbyoc(user_context, model_name, backend_name, target_name, models_dir, feature_names, config):
    df, artifacts = _test_run_platform(
        None, backend_name, target_name, user_context, model_name, models_dir, feature_names, config
    )
    assert "cmsisnnbyoc" in df["Features"][0]
    # TODO: find out if kernels are actually linked?
    # TODO: 2 runs to compare ROM/RAM/Cycles?


@pytest.mark.slow
@pytest.mark.user_context
@pytest.mark.parametrize(
    "model_name", DEFAULT_TFLITE_MODELS
)  # TODO: add test model for this, also test with wrong data
@pytest.mark.parametrize(
    "backend_name", ["tvmaot"]  # -> add tvm if we have a test model for this
)  # TODO: Single backend would be fine, but it has to be enabled
@pytest.mark.parametrize("feature_names", [["unpacked_api"]])
@pytest.mark.parametrize("config", [{}])
def test_feature_unpacked_api(user_context, model_name, backend_name, models_dir, feature_names, config):
    df, artifacts = _test_backend(backend_name, user_context, model_name, models_dir, feature_names, config)
    assert "unpacked_api" in df["Features"][0]
    # TODO: check generated code -> do not run at all (would need to check for metrics changes)


@pytest.mark.slow
@pytest.mark.user_context
@pytest.mark.parametrize(
    "model_name", DEFAULT_TFLITE_MODELS
)  # TODO: add test model for this, also test with wrong data
@pytest.mark.parametrize(
    "backend_name", ["tvmaot"]  # -> add tvm if we have a test model for this
)  # TODO: Single backend would be fine, but it has to be enabled
@pytest.mark.parametrize("feature_names", [["usmp"]])
@pytest.mark.parametrize(
    "config",
    [{"usmp.algorithm": "greedy_by_size"}, {"usmp.algorithm": "greedy_by_conflicts"}, {"usmp.algorithm": "hill_climb"}],
)
def test_feature_usmp(user_context, model_name, backend_name, models_dir, feature_names, config):
    df, artifacts = _test_backend(backend_name, user_context, model_name, models_dir, feature_names, config)
    assert "usmp" in df["Features"][0]
    # TODO: run twice and compare generted code or look for specific code


@pytest.mark.slow
@pytest.mark.user_context
@pytest.mark.parametrize(
    "model_name", DEFAULT_TFLITE_MODELS
)  # TODO: add test model for this, also test with wrong data
@pytest.mark.parametrize(
    "backend_name", ["tvmaot"]  # other tvm backends?
)  # TODO: Single backend would be fine, but it has to be enabled
@pytest.mark.parametrize("feature_names", [["disable_legalize"]])
@pytest.mark.parametrize("config", [{}])
def test_feature_disable_legalize(user_context, model_name, backend_name, models_dir, feature_names, config):
    df, artifacts = _test_backend(backend_name, user_context, model_name, models_dir, feature_names, config)
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
@pytest.mark.parametrize("feature_names", [["autotvm"]])
@pytest.mark.parametrize("config", [{}])
def test_feature_autotvm(
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
    with user_context.create_session() as session:
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
@pytest.mark.parametrize("feature_names", [["autotvm", "autotuned"]])  # TODO: provide tuning records instead
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
    with user_context.create_session() as session:
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


# Platform
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


# Target
# TODO

# TODO:
# cmsisnn -> currently broken
# gdbserver -> hard to test
