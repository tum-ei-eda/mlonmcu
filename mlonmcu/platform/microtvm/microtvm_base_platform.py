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
"""MicroTVM Base Platform"""
import re
import tempfile
from pathlib import Path

# from ..platform import Platform
from ..tvm.tvm_base_platform import TvmBasePlatform
from mlonmcu.setup import utils
from mlonmcu.flow.tvm.backend.python_utils import prepare_python_environment
from mlonmcu.logging import get_logger

logger = get_logger()


def parse_project_options_from_stdout(out):
    return re.compile(r"^\s+([A-Za-z0-9_]+)=", re.MULTILINE).findall(out)


def filter_project_options(valid, options):
    return {key: value for key, value in options.items() if key in valid}


def get_project_option_args(stage, project_options):
    ret = []
    for key, value in project_options.items():
        ret.append(f"{key}={value}")

    if len(ret) > 0:
        ret = ["--project-option"] + ret

    return ret


# TODO: abstarct
class MicroTvmBasePlatform(TvmBasePlatform):
    """MicroTVM base platform class."""

    DEFAULTS = {
        **TvmBasePlatform.DEFAULTS,
        "project_template": None,
        "project_options": {},
        "tvmc_custom_script": None,
        "project_dir": None,
        "experimental_tvmc_micro_tune": False,
        "experimental_tvmc_print_time": False,
    }

    REQUIRED = TvmBasePlatform.REQUIRED | {"tvm.build_dir", "tvm.pythonpath", "tvm.configs_dir"}

    def __init__(self, name, features=None, config=None):
        super().__init__(
            name,
            features=features,
            config=config,
        )
        self.tempdir = None
        self.project_name = "app"
        self.project_dir = None

    def collect_available_project_options(self, command, target=None):
        # TODO: define NotImplemented versions of the invoke_tvmc_micro_* mathods in here
        if "create" in command:
            out = self.invoke_tvmc_micro_create("_", target=target, list_options=True, live=False)
        elif command == "build":
            out = self.invoke_tvmc_micro_build(target=target, list_options=True, live=False)
        elif command == "flash":
            out = self.invoke_tvmc_micro_flash(target=target, list_options=True, live=False)
        elif command == "tune":
            tune_args = ["--output", "-", "_"]
            out = self.invoke_tvmc_micro_tune(*tune_args, target=target, list_options=True, live=False)
        elif command == "run":
            out = self.invoke_tvmc_micro_run(target=target, list_options=True, live=False)
        else:
            raise RuntimeError(f"Unexpected command: {command}")
        return parse_project_options_from_stdout(out)

    def invoke_tvmc_micro(self, command, *args, target=None, list_options=False, **kwargs):
        if list_options:
            project_option_args = ["--help"]
        else:
            options = filter_project_options(
                self.collect_available_project_options(command, target=target),
                target.get_project_options(),
            )
            project_option_args = get_project_option_args(command, options)
        return self.invoke_tvmc("micro", command, *args, *project_option_args, target=target, **kwargs)

    def get_template_args(self, target):
        template = target.template
        if target.template_path:
            template = "template"
            template_path = target.template_path
        else:
            if template == "template":
                assert self.project_template is not None
                template_path = self.project_template
            else:
                template_path = None
        if template_path:
            return ("template", "--template-dir", template_path)
        else:
            return (template,)

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

    @property
    def project_template(self):
        return self.config["project_template"]

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

    def invoke_tvmc(self, command, *args, target=None, live=None, **kwargs):
        env = prepare_python_environment(self.tvm_pythonpath, self.tvm_build_dir, self.tvm_configs_dir)
        if live is None:
            live = self.print_outputs
        if target:
            target.update_environment(env)
        if self.tvmc_custom_script is None:
            pre = ["-m", "tvm.driver.tvmc"]
        else:
            pre = [self.tvmc_custom_script]
        return utils.python(*pre, command, *args, live=live, env=env, **kwargs)

    def close(self):
        if self.tempdir:
            self.tempdir.cleanup()
