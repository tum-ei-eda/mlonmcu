import pytest

from mlonmcu.target.riscv.riscv import RISCVTarget
from mlonmcu.target.riscv.riscv_vext_target import RVVTarget
from mlonmcu.target.riscv.riscv_pext_target import RVPTarget
from mlonmcu.target import register_target
from mlonmcu.environment.config import TargetConfig

from .helpers import MODEL_FRONTENDS, TARGET_PLATFORMS, _check_features, _init_run


def _check_multilib(context, gcc):
    environment = context.environment
    user_vars = environment.vars
    # TODO: properly lookup in dependency cache?
    return user_vars.get(f"{gcc}.multilib", False)


def _get_target_config(target, xlen, fpu, embedded, compressed, atomic, multiply):
    return {
        f"{target}.xlen": xlen,
        f"{target}.embedded": embedded,
        f"{target}.compressed": compressed,
        f"{target}.atomic": atomic,
        f"{target}.multiply": multiply,
        f"{target}.fpu": fpu,
    }


def _get_platform_config(platform, toolchain):
    return {
        f"{platform}.print_outputs": True,
        f"{platform}.toolchain": toolchain,
    }


def _get_feature_vext_config(vlen, elen, spec, embedded):
    return {
        "vext.vlen": vlen,
        "vext.elen": elen,
        "vext.spec": spec,
        "vext.embedded": embedded,
    }


def _get_feature_pext_config(spec):
    return {
        "pext.spec": spec,
    }


class MyRISCVTarget(RISCVTarget):
    """Base RISC-V target with support as we do not run simulations, just compile."""

    def __init__(self, name="myriscv_default", features=None, config=None):
        super().__init__(name, features=features, config=config)


class MyRVVTarget(RVVTarget):
    """RISC-V target with vector (rvv) support as we do not run simulations, just compile."""

    def __init__(self, name="myriscv_vector", features=None, config=None):
        super().__init__(name, features=features, config=config)


class MyRVPTarget(RVPTarget):
    """RISC-V target with packed (rvp) support as we do not run simulations, just compile."""

    def __init__(self, name="myriscv_packed", features=None, config=None):
        super().__init__(name, features=features, config=config)


register_target("myriscv_default", MyRISCVTarget)
register_target("myriscv_vector", MyRVVTarget)
register_target("myriscv_packed", MyRVPTarget)


def _test_compile_platform2(
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
    with session:
        run.add_features_by_name(feature_names, context=user_context)
        run.add_frontend_by_name(frontend_name, context=user_context)
        run.add_model_by_name(model_name, context=user_context)
        run.add_backend_by_name(backend_name, context=user_context)  # TODO: implicit Framework
        run.add_platform_by_name(platform_name, context=user_context)
        run.add_target_by_name(target_name, context=user_context)
        # arch = run.target.arch
        # abi = run.target.abi
        # vext = "vext" in features
        # pext = "pext" in features
        # toolchain = run.platform.toolchain
        # _check_available_multilibs(user_context, arch, abi,... , toolchain)
        # assert session.process_runs(until=RunStage.COMPILE, context=user_context)
    report = session.get_reports()
    df, artifacts = report.df, run.artifacts

    assert len(df) == 1
    assert df["Model"][0] == model_name
    assert df["Platform"][0] == platform_name
    assert df["Target"][0] == target_name
    return df, artifacts


def _test_riscv_toolchain(
    platform_name, backend_name, target_name, user_context, model_name, models_dir, feature_names, config, toolchain
):
    user_context.environment.targets.extend(
        [
            TargetConfig("myriscv_default"),
            TargetConfig("myriscv_vector"),
            TargetConfig("myriscv_packed"),
        ]
    )
    assert platform_name == "mlif"
    user_config = user_context.environment.vars.copy()
    user_config.update(_get_platform_config(platform_name, toolchain))
    if not user_context.environment.has_toolchain(toolchain):
        pytest.skip(f"Toolchain '{toolchain}' not available in user environment!")

    return _test_compile_platform2(
        platform_name, backend_name, target_name, user_context, model_name, models_dir, feature_names, config
    )
