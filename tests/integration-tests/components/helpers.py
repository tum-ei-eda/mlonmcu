import pytest
from mlonmcu.environment.config import PathConfig
from mlonmcu.session.run import RunStage
from mlonmcu.artifact import lookup_artifacts

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

ALL_FRONTENDS = ["tflite", "pb", "relay", "onnx"]

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
