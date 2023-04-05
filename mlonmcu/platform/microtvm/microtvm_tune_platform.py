from mlonmcu.config import str2bool
from .microtvm_target_platform import MicroTvmTargetPlatform
from ..tvm.tvm_tune_platform import TvmTunePlatform

from mlonmcu.flow.tvm.backend.tvmc_utils import (
    get_rpc_tvmc_args,
    get_target_tvmc_args,
)


class MicroTvmTunePlatform(TvmTunePlatform, MicroTvmTargetPlatform):
    """MicroTVM Tune platform class."""

    FEATURES = TvmTunePlatform.FEATURES + MicroTvmTargetPlatform.FEATURES

    DEFAULTS = {
        **TvmTunePlatform.DEFAULTS,
        **MicroTvmTargetPlatform.DEFAULTS,
    }

    REQUIRED = TvmTunePlatform.REQUIRED + MicroTvmTargetPlatform.REQUIRED + []

    @property
    def experimental_tvmc_micro_tune(self):
        value = self.config["experimental_tvmc_micro_tune"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    def invoke_tvmc_micro_tune(self, *args, target=None, list_options=False):
        all_args = []
        all_args.extend(args)
        template_args = self.get_template_args(target)
        all_args.extend(template_args)
        return self.invoke_tvmc_micro("tune", *all_args, target=target, list_options=list_options)

    def invoke_tvmc_tune(self, *args, target=None):
        return self.invoke_tvmc_micro_tune(*args, target=target)

    def _tune_model(self, model_path, backend, target):
        assert self.experimental_tvmc_micro_tune, "Microtvm tuning requires experimental_tvmc_micro_tune"

        return super()._tune_model(model_path, backend, target)
