"""Test used RISC-V GCC Toolchains"""
import pytest

from mlonmcu.environment.config import PathConfig
from mlonmcu.session.run import RunStage
from mlonmcu.target.riscv.riscv import RISCVTarget
from mlonmcu.target.riscv.riscv_vext_target import RVVTarget
from mlonmcu.target.riscv.riscv_pext_target import RVPTarget
from mlonmcu.target import register_target

# Using muRISCV-NN feature to get a relatively good coverage
# It would be nice if we would have specific test models which utilize all relevant kernels
# TODO: also test Zvl* extensions

# MODELS = ["tinymlperf"]
MODELS = ["resnet"]


def _get_target_config(target, xlen, fpu, embedded, compressed, atomic, multiply):
    return {
        f"{target}.xlen": xlen,
        f"{target}.embedded": embedded,
        f"{target}.compressed": compressed,
        f"{target}.atomic": atomic,
        f"{target}.multiply": multiply,
        f"{target}.fpu": fpu,
    }


def _get_platform_config():
    return {
        "mlif.print_outputs": True,
        "mlif.toolchain": "gcc",
    }


def _get_feature_vext_config(vlen, elen, spec, embedded):
    return {
        "vext.vlen": vlen,
        "vext.elen": elen,
        "vext.spec": spec,
        "vext.embedded": embedded,
    }


class MyRISCVTarget(RISCVTarget):
    """Base RISC-V target with support as we do not run simulations, just compile."""

    def __init__(self, name="myriscv_default_gcc", features=None, config=None):
        super().__init__(name, features=features, config=config)


class MyRVVTarget(RVVTarget):
    """RISC-V target with vector (rvv) support as we do not run simulations, just compile."""

    def __init__(self, name="myriscv_vector_gcc", features=None, config=None):
        super().__init__(name, features=features, config=config)


class MyRVPTarget(RVPTarget):
    """RISC-V target with packed (rvp) support as we do not run simulations, just compile."""

    def __init__(self, name="myriscv_packed_gcc", features=None, config=None):
        super().__init__(name, features=features, config=config)


register_target("myriscv_default_gcc", MyRISCVTarget)
register_target("myriscv_vector_gcc", MyRVVTarget)
register_target("myriscv_packed_gcc", MyRVPTarget)


# TODO: share code with test_slow.py
def _check_features(user_context, feature_names):
    for feature in feature_names:
        if not user_context.environment.has_feature(feature):
            pytest.skip(f"Feature '{feature}' is not enabled.")


def _init_run(user_context, models_dir, config):
    user_context.environment.paths["models"] = [PathConfig(models_dir)]
    session = user_context.create_session()
    return session, session.create_run(config=config)


def _test_compile_platform(
    platform_name, backend_name, target_name, user_context, model_name, models_dir, feature_names, config
):
    user_config = user_context.environment.vars.copy()
    user_config.update(config)
    frontend_name = "tflite"
    if not user_context.environment.has_frontend(frontend_name):
        pytest.skip(f"Frontend '{frontend_name}' is not enabled.")
    if not user_context.environment.has_backend(backend_name):
        pytest.skip(f"Backend '{backend_name}' is not enabled.")
    if not user_context.environment.has_platform(platform_name):
        pytest.skip(f"Platform '{platform_name}' is not enabled.")
    # if not user_context.environment.has_target(target_name):
    #     pytest.skip(f"Target '{target_name}' is not enabled.")
    _check_features(user_context, feature_names)
    session, run = _init_run(user_context, models_dir, user_config)
    run.add_features_by_name(feature_names, context=user_context)
    run.add_frontend_by_name(frontend_name, context=user_context)
    run.add_model_by_name(model_name, context=user_context)
    run.add_backend_by_name(backend_name, context=user_context)
    run.add_platform_by_name(platform_name, context=user_context)
    # user_context.environment.targets.append(TargetConfig("myriscv"))
    run.add_target_by_name(target_name, context=user_context)
    assert session.process_runs(until=RunStage.COMPILE, context=user_context)
    report = session.get_reports()
    df, artifacts = report.df, run.artifacts

    assert len(df) == 1
    assert df["Model"][0] == model_name
    assert df["Platform"][0] == platform_name
    assert df["Target"][0] == target_name
    return df, artifacts


@pytest.mark.slow
@pytest.mark.user_context
@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize("target", ["myriscv_default_gcc"])
@pytest.mark.parametrize("xlen", [32, 64])
# @pytest.mark.parametrize("xlen", [32])
# @pytest.mark.parametrize("fpu", ["none", "single", "double"])
@pytest.mark.parametrize("fpu", ["none", "double"])  # single float if often not included in multilib gcc
@pytest.mark.parametrize("embedded,compressed,atomic,multiply", [(False, True, True, True)])
@pytest.mark.parametrize("feature_names", [[], ["muriscvnn"]])
def test_default(
    model_name, target, xlen, fpu, embedded, compressed, atomic, multiply, feature_names, user_context, models_dir
):
    config = {
        **_get_target_config(target, xlen, fpu, embedded, compressed, atomic, multiply),
        **_get_platform_config(),
    }

    _, artifacts = _test_compile_platform(
        "mlif", "tflmi", target, user_context, model_name, models_dir, feature_names, config
    )


@pytest.mark.slow
@pytest.mark.user_context
@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize("target", ["myriscv_default_gcc"])
# @pytest.mark.parametrize("vlen", [64, 128, 256, 512, 1024, 2048])
@pytest.mark.parametrize("vlen", [64, 128, 2048])
@pytest.mark.parametrize("elen", [32])
@pytest.mark.parametrize("spec", [1.0])
# @pytest.mark.parametrize("fpu", ["none", "single", "double"])
@pytest.mark.parametrize("fpu", ["none", "single"])
@pytest.mark.parametrize("embedded,compressed,atomic,multiply", [(False, True, True, True)])
@pytest.mark.parametrize("feature_names", [["vext"], ["vext", "muriscvnn"]])
# @pytest.mark.parametrize("feature_names", [["vext"]])
def test_embedded_vector_32bit(
    model_name,
    target,
    vlen,
    elen,
    spec,
    fpu,
    embedded,
    compressed,
    atomic,
    multiply,
    feature_names,
    user_context,
    models_dir,
):
    if elen == 32:
        pytest.skip(f"ELEN 32 not supported due to compiler bug")

    config = {
        **_get_target_config(target, 32, fpu, embedded, compressed, atomic, multiply),
        **_get_feature_vext_config(vlen, elen, spec, True),
        **_get_platform_config(),
    }

    _, artifacts = _test_compile_platform(
        "mlif", "tflmi", target, user_context, model_name, models_dir, feature_names, config
    )


@pytest.mark.slow
@pytest.mark.user_context
@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize("target", ["myriscv_default_gcc"])
# @pytest.mark.parametrize("vlen", [64, 128, 256, 512, 1024, 2048])
@pytest.mark.parametrize("vlen", [64, 128, 2048])
@pytest.mark.parametrize("elen", [32, 64])
@pytest.mark.parametrize("spec", [1.0])
@pytest.mark.parametrize("fpu", ["none", "single", "double"])
@pytest.mark.parametrize("embedded,compressed,atomic,multiply", [(False, True, True, True)])
@pytest.mark.parametrize("feature_names", [["vext"], ["vext", "muriscvnn"]])
# @pytest.mark.parametrize("feature_names", [["vext"]])
def test_embedded_vector_64bit(
    model_name,
    target,
    vlen,
    elen,
    spec,
    fpu,
    embedded,
    compressed,
    atomic,
    multiply,
    feature_names,
    user_context,
    models_dir,
):
    if elen == 32:
        pytest.skip(f"ELEN 32 not supported due to compiler bug")
    if fpu == "single":
        pytest.skip(f"Single precision float not supported due to compiler bug")

    config = {
        **_get_target_config(target, 64, fpu, embedded, compressed, atomic, multiply),
        **_get_feature_vext_config(vlen, elen, spec, True),
        **_get_platform_config(),
    }

    _, artifacts = _test_compile_platform(
        "mlif", "tflmi", target, user_context, model_name, models_dir, feature_names, config
    )


# @pytest.mark.slow
# @pytest.mark.user_context
# @pytest.mark.parametrize("model_name", MODELS)
# @pytest.mark.parametrize("target", ["myriscv_default_gcc"])
# # @pytest.mark.parametrize("vlen", [64, 128, 256, 512, 1024, 2048])
# @pytest.mark.parametrize("vlen", [64, 128, 2048])
# @pytest.mark.parametrize("spec", [1.0])
# # @pytest.mark.parametrize("fpu", ["none", "single", "double"])
# @pytest.mark.parametrize("embedded,compressed,atomic,multiply", [(False, True, True, True)])
# @pytest.mark.parametrize("feature_names", [["vext"], ["vext", "muriscvnn"]])
# def test_vector_32bit(
#     model_name, target, vlen, spec, fpu, embedded, compressed, atomic, multiply, feature_names, user_context, models_dir
# ):
#     config = {
#         **_get_target_config(target, 32, fpu, embedded, compressed, atomic, multiply),
#         **_get_feature_vext_config(vlen, 64, spec, False),
#         **_get_platform_config(),
#     }
#     _, artifacts = _test_compile_platform(
#         "mlif", "tflmi", target, user_context, model_name, models_dir, feature_names, config
#     )


@pytest.mark.slow
@pytest.mark.user_context
@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize("target", ["myriscv_default_gcc"])
# @pytest.mark.parametrize("vlen", [64, 128, 256, 512, 1024, 2048])
@pytest.mark.parametrize("vlen", [64, 128, 2048])
@pytest.mark.parametrize("spec", [1.0])
# @pytest.mark.parametrize("fpu", ["none", "single", "double"])
@pytest.mark.parametrize("fpu", ["double"])
@pytest.mark.parametrize("embedded,compressed,atomic,multiply", [(False, True, True, True)])
@pytest.mark.parametrize("feature_names", [["vext"], ["vext", "muriscvnn"]])
def test_vector_64bit(
    model_name, target, vlen, spec, fpu, embedded, compressed, atomic, multiply, feature_names, user_context, models_dir
):
    config = {
        **_get_target_config(target, 64, fpu, embedded, compressed, atomic, multiply),
        **_get_feature_vext_config(vlen, 64, spec, False),
        **_get_platform_config(),
    }
    _, artifacts = _test_compile_platform(
        "mlif", "tflmi", target, user_context, model_name, models_dir, feature_names, config
    )


@pytest.mark.slow
@pytest.mark.user_context
@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize("target", ["myriscv_default_gcc"])
@pytest.mark.parametrize("xlen", [32, 64])
@pytest.mark.parametrize("spec", [0.96])
# @pytest.mark.parametrize("fpu", ["none", "single", "double"])
@pytest.mark.parametrize("fpu", ["none", "double"])  # single float if often not included in multilib gcc
@pytest.mark.parametrize("embedded,compressed,atomic,multiply", [(False, True, True, True)])
@pytest.mark.parametrize("feature_names", [["pext"], ["pext", "muriscvnn"]])
def test_packed(
    model_name, target, fpu, embedded, compressed, atomic, multiply, feature_names, user_context, models_dir
):
    config = {
        **_get_target_config(target, 64, fpu, embedded, compressed, atomic, multiply),
        **_get_platform_config(),
    }
    _, artifacts = _test_compile_platform(
        "mlif", "tflmi", target, user_context, model_name, models_dir, feature_names, config
    )


# def test_vector_packed():
