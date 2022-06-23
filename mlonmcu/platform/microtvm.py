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
"""MicroTVM Platform"""

import tempfile
from pathlib import Path

from mlonmcu.setup import utils
from mlonmcu.logging import get_logger
from mlonmcu.target import SUPPORTED_TARGETS
from mlonmcu.target.target import Target
from mlonmcu.artifact import Artifact, ArtifactFormat

from mlonmcu.flow.tvm.backend.python_utils import prepare_python_environment
from mlonmcu.flow.tvm.backend.tvmc_utils import get_bench_tvmc_args, get_data_tvmc_args

from .platform import CompilePlatform, TargetPlatform
from .microtvm_target import create_microtvm_target

logger = get_logger()


# TODO: Replace this hardcoded dict which dynamic lookup
ALLOWED_PROJECT_OPTIONS = {
    "arduino": {
        "create": ["arduino_cli_cmd", "project_type", "warning_as_error"],
        "build": ["arduino_board", "arduino_cli_cmd", "verbose"],
        "flash": ["arduino_board", "arduino_cli_cmd", "port", "verbose"],
        "run": ["arduino_board", "arduino_cli_cmd", "port"],
    },
    "zephyr": {
        "create": [
            "extra_files_tar",
            "project_type",
            "config_main_stack_size",
            "warning_as_error",
            "compile_definitions",
            "zephyr_base",
            "zephyr_board",
        ],
        "build": ["verbose", "west_cmd", "zephyr_base", "zephyr_board"],
        "flash": ["zephyr_board"],
        "run": ["gdbserver_port", "nrfjprog_snr", "openocd_serial", "zephyr_base", "zephyr_board"],
    },
    # "etissvp": {
    "template": {
        "create": [
            "extra_files_tar",
            "project_type",
            "config_main_stack_size",
            "warning_as_error",
            "compile_definitions",
        ],
        "build": ["verbose", "riscv_path", "etiss_path"],
        "flash": ["etiss_path", "etissvp_script", "etissvp_script_args", "transport"],
        "run": ["etissvp_script", "etissvp_script_args", "etiss_path"],
    },
}


def get_project_option_args(template, stage, project_options):
    ret = []
    # TODO: dynamically fetch allowed options per stage (create, build, run)
    assert template[0] in ALLOWED_PROJECT_OPTIONS
    assert stage in ALLOWED_PROJECT_OPTIONS[template[0]]
    allowed = ALLOWED_PROJECT_OPTIONS[template[0]][stage]
    for key, value in project_options.items():
        if key in allowed:
            ret.append(f"{key}={value}")

    if len(ret) > 0:
        ret = ["--project-option"] + ret

    return ret


# TODO: This file is very similar to the TVM platform -> Reuse as much as possible


class MicroTvmPlatform(CompilePlatform, TargetPlatform):
    """TVM Platform class."""

    FEATURES = CompilePlatform.FEATURES + TargetPlatform.FEATURES + ["microtvm_etissvp"]  # TODO: validate?

    DEFAULTS = {
        **CompilePlatform.DEFAULTS,
        **TargetPlatform.DEFAULTS,
        "project_template": None,
        "project_options": {},
        "project_dir": None,
        "fill_mode": "random",
        "ins_file": None,
        "outs_file": None,
        "print_top": False,
        # "profile": False,
        "repeat": 1,
        # "use_rpc": False,
        # "rpc_key": None,
        # "rpc_hostname": None,
        # "rpc_port": None,
        "tvmc_extra_args": [],
        "tvmc_custom_script": None,
    }

    REQUIRED = ["tvm.build_dir", "tvm.pythonpath", "tvm.configs_dir"]

    def __init__(self, features=None, config=None):
        super().__init__(
            "microtvm",  # Actually: tvmllvm
            features=features,
            config=config,
        )
        self.tempdir = None
        self.project_name = "app"
        self.project_dir = None

    @property
    def project_options(self):
        opts = self.config["project_options"]
        if isinstance(opts, str):
            opts_split = opts.split(" ")
            opts = {}
            for opt in opts_split:
                key, value = opt.split("=")[:2]
                opts[key] = value
        assert isinstance(opts, dict)
        return opts

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

    # @property
    # def profile(self):
    #     return str2bool(self.config["profile"])

    @property
    def repeat(self):
        return self.config["repeat"]

    # @property
    # def use_rpc(self):
    #     return str2bool(self.config["use_rpc"])

    # @property
    # def rpc_key(self):
    #     return self.config["rpc_key"]

    # @property
    # def rpc_hostname(self):
    #     return self.config["rpc_hostname"]

    # @property
    # def rpc_port(self):
    #     return self.config["rpc_port"]

    @property
    def tvmc_extra_args(self):
        return self.config["tvmc_extra_args"]

    @property
    def tvmc_custom_script(self):
        return self.config["tvmc_custom_script"]

    @property
    def tvm_pythonpath(self):
        return self.config["tvm.pythonpath"]

    @property
    def tvm_build_dir(self):
        return self.config["tvm.build_dir"]

    @property
    def tvm_configs_dir(self):
        return self.config["tvm.configs_dir"]

    @property
    def project_template(self):
        return self.config["project_template"]

    def init_directory(self, path=None, context=None):
        if self.project_dir is not None:
            self.project_dir.mkdir(exist_ok=True)
            logger.debug("Project directory already initialized")
            return
        dir_name = self.name
        if path is not None:
            self.project_dir = Path(path)
        elif self.config["project_dir"] is not None:
            self.project_dir = Path(self.config["project_dir"])
        else:
            if context:
                assert "temp" in context.environment.paths
                self.project_dir = (
                    context.environment.paths["temp"].path / dir_name
                )  # TODO: Need to lock this for parallel builds
            else:
                logger.debug(
                    "Creating temporary directory because no context was available "
                    "and 'espidf.project_dir' was not supplied"
                )
                self.tempdir = tempfile.TemporaryDirectory()
                self.project_dir = Path(self.tempdir.name) / dir_name
                logger.debug("Temporary project directory: %s", self.project_dir)
        self.project_dir.mkdir(exist_ok=True)

    def get_supported_targets(self):
        # TODO: get this via tvmc micro create-project --help
        target_names = ["zephyr", "arduino", "template"]

        return [f"microtvm_{name}" for name in target_names]

    def create_target(self, name):
        assert name in self.get_supported_targets(), f"{name} is not a valid MicroTVM device"
        if name in SUPPORTED_TARGETS:
            base = SUPPORTED_TARGETS[name]
        else:
            base = Target
        return create_microtvm_target(name, self, base=base)

    def get_tvmc_run_args(self, path, device, num=1):
        return [
            path,
            *["--device", device],
            *get_data_tvmc_args(
                mode=self.fill_mode, ins_file=self.ins_file, outs_file=self.outs_file, print_top=self.print_top
            ),
            *get_bench_tvmc_args(
                # print_time=True, profile=self.profile, end_to_end=False, repeat=self.repeat, number=num
                print_time=True,
                profile=False,
                end_to_end=False,
                repeat=self.repeat,
                number=num,
            ),
            # *get_rpc_tvmc_args(self.use_rpc, self.rpc_key, self.rpc_hostname, self.rpc_port),
        ]

    def get_tvmc_micro_args(self, command, path, mlf_path, template):
        if "create" in command:
            return [command, "--force", path, mlf_path, *template]
        else:
            return [command, path, *template]

    def invoke_tvmc(self, command, *args):
        env = prepare_python_environment(self.tvm_pythonpath, self.tvm_build_dir, self.tvm_configs_dir)
        if self.tvmc_custom_script is None:
            pre = ["-m", "tvm.driver.tvmc"]
        else:
            pre = [self.tvmc_custom_script]
        return utils.python(*pre, command, *args, live=self.print_outputs, print_output=False, env=env)

    def invoke_tvmc_micro(self, command, path, mlf_path, template, micro=True):
        args = self.get_tvmc_micro_args(command, path, mlf_path, template)
        args += get_project_option_args(template, command, self.project_options)
        return self.invoke_tvmc("micro", *args)

    def invoke_tvmc_run(self, path, device, template, num=1, micro=True):
        args = self.get_tvmc_run_args(path, device, num=num)
        if micro:
            args.extend(get_project_option_args(template, "run", self.project_options))
        return self.invoke_tvmc("run", *args)

    def close(self):
        if self.tempdir:
            self.tempdir.cleanup()

    def get_template_args(self, target):
        target_template = target.template
        if target_template == "template":
            assert self.project_template is not None
            return (target_template, "--template-dir", self.project_template)
        else:
            return (target_template,)

    def prepare(self, mlf, target):
        out = self.invoke_tvmc_micro("create", self.project_dir, mlf, self.get_template_args(target))
        return out

    def compile(self, target):
        out = ""
        # TODO: build with cmake options
        out += self.invoke_tvmc_micro("build", self.project_dir, None, self.get_template_args(target))
        # TODO: support self.num_threads (e.g. patch esp-idf)
        return out

    def generate_elf(self, src, target, model=None, num=1, data_file=None):
        # TODO: name missleading as we are not interested in the ELF
        src = Path(src) / "default.tar"  # TODO: lookup for *.tar file
        artifacts = []
        out = self.prepare(src, target)
        out += self.compile(target)
        stdout_artifact = Artifact(
            "microtvm_out.log", content=out, fmt=ArtifactFormat.TEXT  # TODO: split into one file per command
        )
        artifacts.append(stdout_artifact)
        self.artifacts = artifacts

    def flash(self, elf, target, timeout=120):
        # Ignore elf, as we use self.project_dir instead
        # TODO: add alternative approach which allows passing elf instead
        if elf is not None:
            logger.debug("Ignoring ELF file for microtvm platform")
        # TODO: implement timeout
        logger.debug("Flashing target software using MicroTVM ProjectAPI")
        output = self.invoke_tvmc_micro("flash", self.project_dir, None, self.get_template_args(target))
        return output

    def run(self, elf, target, timeout=120, num=1):
        # TODO: implement timeout
        output = self.flash(elf, target)
        output += self.invoke_tvmc_run(
            str(self.project_dir), "micro", self.get_template_args(target), num=num, micro=True
        )
        return output
