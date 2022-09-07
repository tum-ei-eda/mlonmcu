"""Test used RISC-V GCC Toolchains"""
import pytest

from mlonmcu.config import str2bool
from mlonmcu.environment.config import PathConfig
from mlonmcu.session.run import RunStage
from mlonmcu.target.riscv.riscv import RISCVTarget
from mlonmcu.target.riscv.util import update_extensions
from mlonmcu.target import register_target

# Using muRISCV-NN feature to get a relatively good coverage
# It would be nice if we would have specific test models which utilize all relevant kernels
# TODO: also test Zvl* extensions

# MODELS = ["tinymlperf"]
MODELS = ["resnet"]
# EXTENSIONS = (
#     [["i", "m", "a", "c"], ["i", "c"], ["i", "m", "c"], ["i", "m"], ["i", "m", "a"], ["i", "a", "c"], ["i", "a"]],
# )
EXTENSIONS = [["i", "m", "a", "c"]]


class MyRISCVTarget(RISCVTarget):
    """Base RISC-V target with vector+packed support as we do not run simulations, just compile."""

    FEATURES = RISCVTarget.FEATURES + ["vext", "pext"]

    DEFAULTS = {
        **RISCVTarget.DEFAULTS,
        "enable_vext": False,
        "vext_spec": 1.0,
        "embedded_vext": True,
        "enable_pext": False,
        "pext_spec": 0.96,
        "vlen": 0,  # vectorization=off
        "elen": 32,
    }

    def __init__(self, name="myriscv", features=None, config=None):
        super().__init__(name, features=features, config=config)

    @property
    def enable_vext(self):
        value = self.config["enable_vext"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def enable_pext(self):
        value = self.config["enable_pext"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def extensions(self):
        exts = super().extensions
        return update_extensions(
            exts,
            pext=self.enable_pext,
            pext_spec=self.pext_spec,
            vext=self.enable_vext,
            elen=self.elen,
            embedded=self.embedded_vext,
            fpu=self.fpu,
            variant=self.gcc_variant,
        )

    @property
    def vlen(self):
        return int(self.config["vlen"])

    @property
    def elen(self):
        return int(self.config["elen"])

    @property
    def vext_spec(self):
        return self.config["vext_spec"]

    @property
    def embedded_vext(self):
        # No str2bool here as this is only used for tests
        return bool(self.config["embedded_vext"])

    @property
    def pext_spec(self):
        return self.config["pext_spec"]

    def exec(self, program, *args, cwd=None, **kwargs):
        raise NotImplementedError

    def get_platform_defs(self, platform):
        ret = super().get_platform_defs(platform)
        if self.enable_pext:
            major, minor = str(float(self.pext_spec)).split(".")[:2]
            ret["RISCV_RVP_MAJOR"] = major
            ret["RISCV_RVP_MINOR"] = minor
        if self.enable_vext:
            major, minor = str(float(self.vext_spec)).split(".")[:2]
            ret["RISCV_RVV_MAJOR"] = major
            ret["RISCV_RVV_MINOR"] = minor
            ret["RISCV_RVV_VLEN"] = self.vlen
        return ret


register_target("myriscv", MyRISCVTarget)
# TODO: pass by cls instead of name


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
@pytest.mark.parametrize("xlen", [32, 64])
# @pytest.mark.parametrize("fpu", ["none", "single", "double"])
@pytest.mark.parametrize("fpu", ["none", "double"])  # single float if often not included in multilib gcc
@pytest.mark.parametrize("extensions", EXTENSIONS)
@pytest.mark.parametrize("feature_names", [[], ["muriscvnn"]])
def test_default(model_name, xlen, fpu, extensions, feature_names, user_context, models_dir):
    # config = {"myriscv.xlen": xlen, "myriscv.extensions": extensions, "myriscv.fpu": fpu}
    config = {"myriscv.xlen": xlen, "myriscv.extensions": extensions, "myriscv.fpu": fpu, "mlif.print_outputs": True}
    _, artifacts = _test_compile_platform(
        "mlif", "tflmi", "myriscv", user_context, model_name, models_dir, feature_names, config
    )


@pytest.mark.slow
@pytest.mark.user_context
@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize("xlen", [32, 64])
# @pytest.mark.parametrize("vlen", [64, 128, 256, 512, 1024, 2048])
@pytest.mark.parametrize("vlen", [64, 128, 2048])
@pytest.mark.parametrize("elen", [32, 64])
@pytest.mark.parametrize("spec", [1.0])
@pytest.mark.parametrize("embedded", [False, True])
# @pytest.mark.parametrize("fpu", ["none", "single", "double"])
@pytest.mark.parametrize("fpu", ["none", "double"])
@pytest.mark.parametrize("extensions", EXTENSIONS)
@pytest.mark.parametrize("feature_names", [["vext"], ["vext", "muriscvnn"]])
def test_vector(model_name, xlen, vlen, elen, spec, embedded, fpu, extensions, feature_names, user_context, models_dir):
    if embedded:
        pytest.skip("Embedded vector extension is currently not supported by default toolchain")
    if (fpu == "double" and xlen == 32 and embedded) or (not embedded and fpu != "double"):
        pytest.skip("Unsupported combination")
    config = {
        "myriscv.xlen": xlen,
        "myriscv.extensions": extensions,
        "myriscv.fpu": fpu,
        "vext.vlen": vlen,
        "vext.elen": elen,
        "vext.spec": spec,
        "vext.embedded": embedded,
    }
    _, artifacts = _test_compile_platform(
        "mlif", "tflmi", "myriscv", user_context, model_name, models_dir, feature_names, config
    )


@pytest.mark.slow
@pytest.mark.user_context
@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize("xlen", [32, 64])
@pytest.mark.parametrize("spec", [0.96])
# @pytest.mark.parametrize("fpu", ["none", "single", "double"])
@pytest.mark.parametrize("fpu", ["none", "double"])  # single float if often not included in multilib gcc
@pytest.mark.parametrize("extensions", EXTENSIONS)
@pytest.mark.parametrize("feature_names", [["pext"], ["pext", "muriscvnn"]])
def test_packed(model_name, xlen, spec, fpu, extensions, feature_names, user_context, models_dir):
    config = {
        "myriscv.xlen": xlen,
        "myriscv.extensions": extensions,
        "myriscv.fpu": fpu,
        "pext.spec": spec,
    }
    _, artifacts = _test_compile_platform(
        "mlif", "tflmi", "myriscv", user_context, model_name, models_dir, feature_names, config
    )


# def test_vector_packed():
