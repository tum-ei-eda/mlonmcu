#
# Copyright (c) 2022 TUM Department of Electrical and Computer Engineering.
#
# This file is part of MLonMCU.
# See https://github.com/tum-ei-eda/mlonmcu.git for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""MicroTVM Target Platform"""
from mlonmcu.config import str2bool
from ..tvm.tvm_target_platform import TvmTargetPlatform
from .microtvm_base_platform import (
    filter_project_options,
    get_project_option_args,
)  # TODO: move to utils
from mlonmcu.flow.tvm.backend.tvmc_utils import (
    get_bench_tvmc_args,
    get_data_tvmc_args,
    # get_rpc_tvmc_args,
)
from mlonmcu.logging import get_logger

logger = get_logger()


class MicroTvmTargetPlatform(TvmTargetPlatform):
    """MicroTVM target platform class."""

    FEATURES = (
        # TvmTargetPlatform.FEATURES +
        set()
        # Warning: benchamrk and profile not supported!
    )

    DEFAULTS = {
        **TvmTargetPlatform.DEFAULTS,
        "experimental_tvmc_print_time": False,
        "skip_flash": False,
        # Warning: contains configs not supported by microtvm
    }

    @property
    def experimental_tvmc_print_time(self):
        value = self.config["experimental_tvmc_print_time"]
        return str2bool(value)

    @property
    def skip_flash(self):
        value = self.config["skip_flash"]
        return str2bool(value)

    def invoke_tvmc_micro_flash(self, target=None, list_options=False, **kwargs):
        all_args = []
        all_args.append(self.project_dir)
        template_args = self.get_template_args(target)
        all_args.extend(template_args)
        return self.invoke_tvmc_micro("flash", *all_args, target=target, list_options=list_options, **kwargs)

    def invoke_tvmc_micro_run(self, *args, target=None, list_options=False, **kwargs):
        all_args = []
        all_args.extend(args)
        all_args.append(self.project_dir)
        all_args.extend(["--device", "micro"])
        if list_options:
            all_args.append("--list-options")
        else:
            options = filter_project_options(
                self.collect_available_project_options("run", target=target), target.get_project_options()
            )
            all_args.extend(get_project_option_args("run", options))
        return self.invoke_tvmc("run", *all_args, target=target, **kwargs)

    def invoke_tvmc_run(self, *args, target=None):
        return self.invoke_tvmc_micro_run(*args, target=target)

    def get_tvmc_run_args(self):
        if self.use_rpc:
            raise RuntimeError("RPC is only supported for tuning with microtvm platform")
        if self.profile:
            assert (
                self.experimental_tvmc_print_time
            ), "MicroTVM profiling is only supported in environments with microtvm.experimental_tvmc_print_time=1"
        ret = [
            *get_data_tvmc_args(
                mode=self.fill_mode, ins_file=self.ins_file, outs_file=self.outs_file, print_top=self.print_top
            ),
            *get_bench_tvmc_args(
                print_time=self.experimental_tvmc_print_time and not self.profile,
                profile=self.profile and self.experimental_tvmc_print_time,
                end_to_end=False,
                repeat=self.repeat if self.experimental_tvmc_print_time else None,
                number=self.number if self.experimental_tvmc_print_time else None,
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
        output = ""
        if not self.skip_flash:
            output += self.flash(elf, target)
        run_args = self.get_tvmc_run_args()
        output += self.invoke_tvmc_run(*run_args, target=target)

        return output
