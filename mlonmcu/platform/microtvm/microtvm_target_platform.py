from mlonmcu.config import str2bool
from .microtvm_rpc_platform import MicroTvmRpcPlatform
from .microtvm_base_platform import (
    filter_project_options,
    get_project_option_args,
)  # TODO: move to utils
from ..platform import TargetPlatform
from mlonmcu.flow.tvm.backend.tvmc_utils import (
    get_bench_tvmc_args,
    get_data_tvmc_args,
    # get_rpc_tvmc_args,
)
from mlonmcu.logging import get_logger

logger = get_logger()


class MicroTvmTargetPlatform(TargetPlatform, MicroTvmRpcPlatform):
    """MicroTVM target platform class."""

    FEATURES = (
        TargetPlatform.FEATURES
        + MicroTvmRpcPlatform.FEATURES
        + [
            "benchmark",
            "tvm_profile",
        ]
    )

    DEFAULTS = {
        **TargetPlatform.DEFAULTS,
        **MicroTvmRpcPlatform.DEFAULTS,
        "fill_mode": "random",
        "ins_file": None,
        "outs_file": None,
        "print_top": False,
        "profile": False,
        "repeat": 1,
        "number": 1,
        "aggregate": "none",  # Allowed: avg, max, min, none, all
        "total_time": False,
    }

    REQUIRED = TargetPlatform.REQUIRED + MicroTvmRpcPlatform.REQUIRED + []

    @property
    def fill_mode(self):
        return self.config["fill_mode"]

    @property
    def ins_file(self):
        return self.config["ins_file"]

    @property
    def outs_file(self):
        return self.config["outs_file"]

    @property
    def print_top(self):
        return self.config["print_top"]

    @property
    def profile(self):
        value = self.config["profile"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def repeat(self):
        return self.config["repeat"]

    @property
    def number(self):
        return self.config["number"]

    @property
    def aggregate(self):
        value = self.config["aggregate"]
        assert value in ["avg", "all", "max", "min", "none"]
        return value

    @property
    def total_time(self):
        value = self.config["total_time"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    def invoke_tvmc_micro_flash(self, target=None, list_options=False):
        all_args = []
        all_args.append(self.project_dir)
        template_args = self.get_template_args(target)
        all_args.extend(template_args)
        return self.invoke_tvmc_micro("flash", *all_args, target=target, list_options=list_options)

    def invoke_tvmc_micro_run(self, *args, target=None, list_options=False):
        all_args = []
        all_args.append(self.project_dir)
        all_args.extend(["--device", "micro"])
        if list_options:
            all_args.append("--list-options")
        else:
            options = filter_project_options(
                self.collect_available_project_options("run", target=target), target.get_project_options()
            )
            all_args.extend(get_project_option_args("run", options))
        return self.invoke_tvmc("run", *all_args, target=target)

    def invoke_tvmc_run(self, *args, target=None):
        return self.invoke_tvmc_micro_run(*args, target=target)

    def get_tvmc_run_args(self):
        if self.use_rpc:
            raise RuntimeError("RPC is only supported for tuning with microtvm platform")
        if self.profile:
            assert (
                self.experimental_tvmc_print_time
            ), "MicroTVM profiloing is only supported in environments with microtvm.experimental_tvmc_print_time=  1"
        ret = [
            *get_data_tvmc_args(
                mode=self.fill_mode, ins_file=self.ins_file, outs_file=self.outs_file, print_top=self.print_top
            ),
            *get_bench_tvmc_args(
                print_time=self.experimental_tvmc_print_time and not self.profile,
                profile=self.profile and self.experimental_tvmc_print_time,
                end_to_end=False,
                # repeat=self.repeat if self.experimental_tvmc_print_time else None,
                # number=self.number if self.experimental_tvmc_print_time else None,
            ),
            # *get_rpc_tvmc_args(self.use_rpc, self.rpc_key, self.rpc_hostname, self.rpc_port),
        ]
        return ret

    def flash(self, elf, target, timeout=120):
        # Ignore elf, as we use self.project_dir instead
        # TODO: add alternative approach which allows passing elf instead
        if elf is not None:
            logger.debug("Ignoring ELF file for microtvm platform")
        # TODO: implement timeout
        logger.debug("Flashing target software using MicroTVM ProjectAPI")
        output = self.invoke_tvmc_micro_flash(target=target)
        return output

    def monitor(self, target, timeout=60):
        raise NotImplementedError

    def run(self, elf, target, timeout=120):
        # TODO: implement timeout
        output = self.flash(elf, target)
        run_args = self.get_tvmc_run_args()
        output += self.invoke_tvmc_run(*run_args, target=target)

        return output
