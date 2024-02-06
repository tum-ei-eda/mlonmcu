"""Test used RISC-V GCC Toolchains"""

# import pytest

# from mlonmcu.environment.config import PathConfig
# from mlonmcu.session.run import RunStage
# from mlonmcu.target.riscv.riscv import RISCVTarget
# from mlonmcu.target.riscv.riscv_vext_target import RVVTarget
# from mlonmcu.target.riscv.riscv_pext_target import RVPTarget
# from mlonmcu.target import register_target

# Using muRISCV-NN feature to get a relatively good coverage
# It would be nice if we would have specific test models which utilize all relevant kernels
# TODO: also test Zvl* extensions

# MODELS = ["tinymlperf"]
MODELS = ["resnet"]


# @pytest.mark.slow
# @pytest.mark.user_context
# @pytest.mark.needs
# @pytest.mark.parametrize("model_name", MODELS)
# @pytest.mark.parametrize("target", ["myriscv_default_llvm"])
# @pytest.mark.parametrize("xlen", [32, 64])
# # @pytest.mark.parametrize("fpu", ["none", "single", "double"])
# @pytest.mark.parametrize("fpu", ["none", "single", "double"])
# @pytest.mark.parametrize("embedded,compressed,atomic,multiply", [(False, True, True, True)])
# @pytest.mark.parametrize("feature_names", [[], ["muriscvnn"]])
# def test_default(
#     model_name, target, xlen, fpu, embedded, compressed, atomic, multiply, feature_names, user_context, models_dir
# ):
#     has_multilib = _check_multilib(user_context, "riscv_gcc")
#     if xlen == 64 and not has_multilib:
#         pytest.skip(f"XLEN=64 needs multilib gcc")
#     if (not compressed or not atomic or not multiply) and not has_multilib:
#         pytest.skip(f"Non-multilib gcc only supports requires: c+m+a")
#     if embedded and not has_multilib:
#         pytest.skip(f"Embedded extension not supported by non-multilib gcc")
#     if fpu in ["none", "single"] and not has_multilib:
#         pytest.skip(f"Non-multilib gcc only supports double extension")
#
#     config = {
#         **_get_target_config(target, xlen, fpu, embedded, compressed, atomic, multiply),
#         **_get_platform_config(),
#     }
#
#     _, artifacts = _test_compile_platform(
#         "mlif", "tflmi", target, user_context, model_name, models_dir, feature_names, config
#     )
#
#
# @pytest.mark.slow
# @pytest.mark.user_context
# @pytest.mark.parametrize("model_name", MODELS)
# @pytest.mark.parametrize("target", ["myriscv_vector_llvm"])
# # @pytest.mark.parametrize("vlen", [64, 128, 256, 512, 1024, 2048])
# @pytest.mark.parametrize("vlen", [64, 128, 2048])
# @pytest.mark.parametrize("elen", [32])
# @pytest.mark.parametrize("spec", [1.0])
# # @pytest.mark.parametrize("fpu", ["none", "single", "double"])
# @pytest.mark.parametrize("fpu", ["none", "single"])
# @pytest.mark.parametrize("embedded,compressed,atomic,multiply", [(False, True, True, True)])
# @pytest.mark.parametrize("feature_names", [["vext"], ["vext", "muriscvnn"]])
# def test_embedded_vector_32bit(
#     model_name,
#     target,
#     vlen,
#     elen,
#     spec,
#     fpu,
#     embedded,
#     compressed,
#     atomic,
#     multiply,
#     feature_names,
#     user_context,
#     models_dir,
# ):
#     has_multilib = _check_multilib(user_context, "riscv_gcc_vext")
#     if (not compressed or not atomic or not multiply) and not has_multilib:
#         pytest.skip(f"Non-multilib gcc only supports requires: c+m+a")
#     if embedded and not has_multilib:
#         pytest.skip(f"Embedded extension not supported by non-multilib gcc")
#     if fpu in ["none", "single"] and not has_multilib:
#         pytest.skip(f"Non-multilib gcc only supports double extension")
#
#     config = {
#         **_get_target_config(target, 32, fpu, embedded, compressed, atomic, multiply),
#         **_get_feature_vext_config(vlen, elen, spec, True),
#         **_get_platform_config(),
#     }
#
#     _, artifacts = _test_compile_platform(
#         "mlif", "tflmi", target, user_context, model_name, models_dir, feature_names, config
#     )
#     _, artifacts = _test_compile_platform(
#         "mlif", "tflmi", target, user_context, model_name, models_dir, feature_names, config
#     )
#
#
# @pytest.mark.slow
# @pytest.mark.user_context
# @pytest.mark.parametrize("model_name", MODELS)
# @pytest.mark.parametrize("target", ["myriscv_vector_llvm"])
# # @pytest.mark.parametrize("vlen", [64, 128, 256, 512, 1024, 2048])
# @pytest.mark.parametrize("vlen", [64, 128, 2048])
# @pytest.mark.parametrize("elen", [32, 64])
# @pytest.mark.parametrize("spec", [1.0])
# @pytest.mark.parametrize("fpu", ["none", "single", "double"])
# @pytest.mark.parametrize("embedded,compressed,atomic,multiply", [(False, True, True, True)])
# @pytest.mark.parametrize("feature_names", [["vext"], ["vext", "muriscvnn"]])
# def test_embedded_vector_64bit(
#     model_name,
#     target,
#     vlen,
#     elen,
#     spec,
#     fpu,
#     embedded,
#     compressed,
#     atomic,
#     multiply,
#     feature_names,
#     user_context,
#     models_dir,
# ):
#     has_multilib = _check_multilib(user_context, "riscv_gcc_vext")
#     if (not compressed or not atomic or not multiply) and not has_multilib:
#         pytest.skip(f"Non-multilib gcc only supports requires: c+m+a")
#     if embedded and not has_multilib:
#         pytest.skip(f"Embedded extension not supported by non-multilib gcc")
#
#     if elen == 32:
#         pytest.skip(f"ELEN 32 not supported due to compiler bug")
#     if fpu == "single":
#         pytest.skip(f"Single precision float not supported due to compiler bug")
#
#     config = {
#         **_get_target_config(target, 64, fpu, embedded, compressed, atomic, multiply),
#         **_get_feature_vext_config(vlen, elen, spec, True),
#         **_get_platform_config(),
#     }
#     _, artifacts = _test_compile_platform(
#         "mlif", "tflmi", target, user_context, model_name, models_dir, feature_names, config
#     )
#
#
# # @pytest.mark.slow
# # @pytest.mark.user_context
# # @pytest.mark.parametrize("model_name", MODELS)
# # @pytest.mark.parametrize("target", ["myriscv_vector_llvm"])
# # # @pytest.mark.parametrize("vlen", [64, 128, 256, 512, 1024, 2048])
# # @pytest.mark.parametrize("vlen", [64, 128, 2048])
# # @pytest.mark.parametrize("spec", [1.0])
# # # @pytest.mark.parametrize("fpu", ["none", "single", "double"])
# # @pytest.mark.parametrize("embedded,compressed,atomic,multiply", [(False, True, True, True)])
# # @pytest.mark.parametrize("feature_names", [["vext"], ["vext", "muriscvnn"]])
# # def test_vector_32bit(model_name, target, vlen, spec, embedded, compressed,
#       atomic, multiply, feature_names, user_context, models_dir):
# #     config = {
# #         **get_target_config(target, 64, fpu, embedded, compressed, atomic, multiply),
# #         **get_feature_vext_config(vlen, 64, spec, True),
# #         **get_platform_config(),
# #     }
# #     _, artifacts = _test_compile_platform(
# #         "mlif", "tflmi", target, user_context, model_name, models_dir, feature_names, config
# #     )
#
#
# @pytest.mark.slow
# @pytest.mark.user_context
# @pytest.mark.parametrize("model_name", MODELS)
# @pytest.mark.parametrize("target", ["myriscv_vector_llvm"])
# # @pytest.mark.parametrize("vlen", [64, 128, 256, 512, 1024, 2048])
# @pytest.mark.parametrize("vlen", [64, 128, 2048])
# @pytest.mark.parametrize("spec", [1.0])
# # @pytest.mark.parametrize("fpu", ["none", "single", "double"])
# @pytest.mark.parametrize("fpu", ["double"])
# @pytest.mark.parametrize("embedded,compressed,atomic,multiply", [(False, True, True, True)])
# @pytest.mark.parametrize("feature_names", [["vext"], ["vext", "muriscvnn"]])
# def test_vector_64bit(
#     model_name, target, vlen, spec, fpu, embedded, compressed, atomic, multiply,
#     feature_names, user_context, models_dir
# ):
#     has_multilib = _check_multilib(user_context, "riscv_gcc_vext")
#     if (not compressed or not atomic or not multiply) and not has_multilib:
#         pytest.skip(f"Non-multilib gcc only supports requires: c+m+a")
#     if embedded and not has_multilib:
#         pytest.skip(f"Embedded extension not supported by non-multilib gcc")
#
#     config = {
#         **_get_target_config(target, 64, fpu, embedded, compressed, atomic, multiply),
#         **_get_feature_vext_config(vlen, 64, spec, False),
#         **_get_platform_config(),
#     }
#     _, artifacts = _test_compile_platform(
#         "mlif", "tflmi", target, user_context, model_name, models_dir, feature_names, config
#     )
