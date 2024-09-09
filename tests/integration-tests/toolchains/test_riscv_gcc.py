"""Test used RISC-V GCC Toolchains"""

import pytest

# from mlonmcu.session.run import RunStage
from mlonmcu.testing.riscv_toolchain import (
    _get_target_config,
    _get_feature_vext_config,
    _get_feature_pext_config,
    _test_riscv_toolchain,
)

# Using muRISCV-NN feature to get a relatively good coverage
# It would be nice if we would have specific test models which utilize all relevant kernels
# TODO: also test Zvl* extensions

# MODELS = ["tinymlperf"]
MODELS = ["resnet"]


@pytest.mark.slow
@pytest.mark.user_context
@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize("target", ["myriscv_default"])
@pytest.mark.parametrize("xlen", [32, 64])
# @pytest.mark.parametrize("xlen", [32])
@pytest.mark.parametrize("fpu", ["none", "single", "double"])
@pytest.mark.parametrize("embedded,compressed,atomic,multiply", [(False, True, True, True)])
@pytest.mark.parametrize("feature_names", [[], ["muriscvnn"]])
def test_default(
    model_name, target, xlen, fpu, embedded, compressed, atomic, multiply, feature_names, user_context, models_dir
):
    config = {
        **_get_target_config(target, xlen, fpu, embedded, compressed, atomic, multiply),
    }

    _, artifacts = _test_riscv_toolchain(
        "mlif", "tflmi", target, user_context, model_name, models_dir, feature_names, config, "gcc"
    )


@pytest.mark.slow
@pytest.mark.user_context
@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize("target", ["myriscv_vector"])
# @pytest.mark.parametrize("vlen", [64, 128, 256, 512, 1024, 2048])
@pytest.mark.parametrize("vlen", [64, 128, 2048])
@pytest.mark.parametrize("elen", [32])
@pytest.mark.parametrize("spec", [1.0])
@pytest.mark.parametrize("fpu", ["none", "single", "double"])
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
        pytest.skip("ELEN 32 not supported due to compiler bug")

    config = {
        **_get_target_config(target, 32, fpu, embedded, compressed, atomic, multiply),
        **_get_feature_vext_config(vlen, elen, spec, True),
    }

    _, artifacts = _test_riscv_toolchain(
        "mlif", "tflmi", target, user_context, model_name, models_dir, feature_names, config, "gcc"
    )


@pytest.mark.slow
@pytest.mark.user_context
@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize("target", ["myriscv_vector"])
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
        pytest.skip("ELEN 32 not supported due to compiler bug")
    if fpu == "single":
        pytest.skip("Single precision float not supported due to compiler bug")

    config = {
        **_get_target_config(target, 64, fpu, embedded, compressed, atomic, multiply),
        **_get_feature_vext_config(vlen, elen, spec, True),
    }

    _, artifacts = _test_riscv_toolchain(
        "mlif",
        "tflmi",
        target,
        user_context,
        model_name,
        models_dir,
        feature_names,
        config,
        "gcc",
    )


# @pytest.mark.slow
# @pytest.mark.user_context
# @pytest.mark.parametrize("model_name", MODELS)
# @pytest.mark.parametrize("target", ["myriscv_vector"])
# # @pytest.mark.parametrize("vlen", [64, 128, 256, 512, 1024, 2048])
# @pytest.mark.parametrize("vlen", [64, 128, 2048])
# @pytest.mark.parametrize("spec", [1.0])
# # @pytest.mark.parametrize("fpu", ["none", "single", "double"])
# @pytest.mark.parametrize("embedded,compressed,atomic,multiply", [(False, True, True, True)])
# @pytest.mark.parametrize("feature_names", [["vext"], ["vext", "muriscvnn"]])
# def test_vector_32bit(
#     model_name, target, vlen, spec, fpu, embedded, compressed, atomic,
#     multiply, feature_names, user_context, models_dir
# ):
#     config = {
#         **_get_target_config(target, 32, fpu, embedded, compressed, atomic, multiply),
#         **_get_feature_vext_config(vlen, 64, spec, False),
#         **_get_platform_config(),
#     }
#     _, artifacts = _test_riscv_toolchain(
#         "mlif", "tflmi", target, user_context, model_name, models_dir, feature_names, config, "gcc"
#     )


@pytest.mark.slow
@pytest.mark.user_context
@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize("target", ["myriscv_vector"])
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
    }
    _, artifacts = _test_riscv_toolchain(
        "mlif", "tflmi", target, user_context, model_name, models_dir, feature_names, config, "gcc"
    )


@pytest.mark.slow
@pytest.mark.user_context
@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize("target", ["myriscv_packed"])
@pytest.mark.parametrize("xlen", [32, 64])
@pytest.mark.parametrize("spec", [0.96])
@pytest.mark.parametrize("fpu", ["none", "single", "double"])
@pytest.mark.parametrize("embedded,compressed,atomic,multiply", [(False, True, True, True)])
@pytest.mark.parametrize("feature_names", [["pext"], ["pext", "muriscvnn"]])
def test_packed(
    model_name, target, xlen, spec, fpu, embedded, compressed, atomic, multiply, feature_names, user_context, models_dir
):
    config = {
        **_get_target_config(target, xlen, fpu, embedded, compressed, atomic, multiply),
        **_get_feature_pext_config(spec),
    }
    _, artifacts = _test_riscv_toolchain(
        "mlif", "tflmi", target, user_context, model_name, models_dir, feature_names, config, "gcc"
    )


# def test_vector_packed():
