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
"""CFU Playground Platform"""

import os
import shutil
import tempfile
import subprocess
from pathlib import Path
import pkg_resources


from mlonmcu.setup import utils
from mlonmcu.artifact import Artifact, ArtifactFormat
from mlonmcu.logging import get_logger
from mlonmcu.target.target import Target
from mlonmcu.config import str2bool, str2list, str2dict

from ..platform import CompilePlatform, TargetPlatform
from .cfu_playground_target import create_cfu_playground_platform_target, get_cfu_playground_platform_targets

logger = get_logger()


def get_project_template(name="proj_template_no_tflm"):
    cfu_templates = pkg_resources.resource_listdir("mlonmcu", os.path.join("resources", "platforms", "cfu_playground"))
    if name not in cfu_templates:
        return None
    fname = pkg_resources.resource_filename("mlonmcu", os.path.join("resources", "platforms", "cfu_playground", name))
    return fname


class CFUPlaygroundPlatform(CompilePlatform, TargetPlatform):
    """CFU Playground Platform class."""

    FEATURES = CompilePlatform.FEATURES | TargetPlatform.FEATURES | {"benchmark"}

    DEFAULTS = {
        **CompilePlatform.DEFAULTS,
        **TargetPlatform.DEFAULTS,
        "project_template": None,
        "project_dir": None,
        "optimize": None,  # values: 0,2,s (s implies z for llvm) only!
        # "device": "digilent_arty",
        # "use_renode": True,
        # "use_verilator": True,
    }

    REQUIRED = {"cfu_playground.src_dir", "yosys.install_dir"}  # TODO: yosys, riscv tc?

    def __init__(self, features=None, config=None):
        super().__init__(
            "cfu_playground",
            features=features,
            config=config,
        )
        self.tempdir = None
        self.project_name = "app"
        self.project_dir = None

    @property
    def cfu_playground_src_dir(self):
        return Path(self.config["cfu_playground.src_dir"])

    @property
    def yosys_install_dir(self):
        return Path(self.config["yosys.install_dir"])

    @property
    def use_renode(self):
        return True

    def init_directory(self, path=None, context=None):
        if self.project_dir is not None:
            self.project_dir.mkdir(exist_ok=True)
            logger.debug("Project directory already initialized")
            return self.project_dir
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
                    "and 'cfu_playground.project_dir' was not supplied"
                )
                self.tempdir = tempfile.TemporaryDirectory()
                self.project_dir = Path(self.tempdir.name) / dir_name
                logger.debug("Temporary project directory: %s", self.project_dir)
        self.project_dir.mkdir(exist_ok=True)
        return self.project_dir

    def _get_supported_targets(self):
        print("_get_supported_targets")
        ret = get_cfu_playground_platform_targets()
        print("ret", ret)
        return ret

    def create_target(self, name):
        supported = self.get_supported_targets()
        assert name in supported, f"{name} is not a valid CFU Playground device"
        base = supported[name]
        return create_cfu_playground_platform_target(name, self, base=base)

    @property
    def project_template(self):
        return self.config["project_template"]

    @property
    def port(self):
        return self.config["port"]

    @property
    def baud(self):
        return self.config["baud"]

    @property
    def optimize(self):
        val = self.config["optimize"]
        if val is None:
            val = "s"
        else:
            val = str(val)
        assert val in ["0", "g", "2", "s", "z"], f"Unsupported: {val}"
        return val

    def close(self):
        if self.tempdir:
            self.tempdir.cleanup()

    def check(self):
        pass

    def prepare_environment(self, target=None):
        env = os.environ.copy()
        # TODO: riscv tc from target
        # TODO: yosys
        # TODO: ...
        env["CFU_ROOT"] = self.cfu_playground_src_dir
        env["PROJ"] = "cfu_playground"
        env["PROJ_DIR"] = self.project_dir
        new_path = env.get("PATH", "")
        # new_path = f"{self.yosys_install_dir}/bin:{old_path}"
        new_path = f"{self.yosys_install_dir}:{new_path}"
        if target:
            new_path = f"{target.riscv_gcc_prefix}/bin:{new_path}"
        print("new_path", new_path)
        env["PATH"] = new_path
        return env

    def prepare(self, target, src):
        self.init_directory()
        self.check()
        template_dir = self.project_template
        if template_dir is None:
            template_dir = get_project_template()
        else:
            template_dir = Path(template_dir)
            if not template_dir.is_dir():
                template_dir = get_project_template(name=str(template_dir))
        assert template_dir is not None, f"Provided project template does not exists: {template_dir}"
        print("cp", template_dir, self.project_dir)
        shutil.copytree(template_dir, self.project_dir, dirs_exist_ok=True)
        # TODO: call make (without renode)?
        out = ""

        return out

    def compile(self, target, src=None):
        # TODO: call make (without renode)?
        out = self.prepare(target, src=src)
        out += utils.make(
            "software", cwd=self.project_dir, env=self.prepare_environment(target=target), live=self.print_outputs
        )
        return out

    def generate(self, src, target, model=None):
        # TODO: implement
        artifacts = []
        out = self.compile(target, src=src)
        elf_name = "software.elf"
        elf_file = self.project_dir / "build" / elf_name
        # TODO: just use path instead of raw data?
        if self.tempdir:
            # Warning: The porject context will get destroyed afterwards wehen using  a temporory directory
            with open(elf_file, "rb") as handle:
                data = handle.read()
                artifact = Artifact("generic_mlif", raw=data, fmt=ArtifactFormat.RAW)
                artifacts.append(artifact)
        else:
            artifact = Artifact(elf_name, path=elf_file, fmt=ArtifactFormat.PATH)
            artifacts.append(artifact)
        metrics = self.get_metrics(elf_file)
        stdout_artifact = Artifact(
            "cfu_playground_out.log", content=out, fmt=ArtifactFormat.TEXT  # TODO: split into one file per command
        )  # TODO: rename to tvmaot_out.log?
        artifacts.append(stdout_artifact)
        return {"default": artifacts}, {"default": metrics}

    def flash(self, elf, target, timeout=120):
        # Ignore elf, as we use self.project_dir instead
        # TODO: add alternative approach which allows passing elf instead
        if elf is not None:
            logger.debug("Ignoring ELF file for cfu platform")
        # TODO: implement timeout
        # TODO: make sure that already compiled? -> error or just call compile routine?
        if self.use_renode:
            pass
        else:
            raise NotImplementedError("Only renode is supported")
            if self.wait_for_user:  # INTERACTIVE
                answer = input(
                    f"Make sure that the device '{target.name}' is connected before you press [Enter]"
                    + " (Type 'Abort' to cancel)"
                )
                if answer.lower() == "abort":
                    return ""
            logger.debug("Flashing target software")

    def monitor(self, target, timeout=60):
        # if self.flash_only:
        #     return ""
        # TODO: make renode or FPGA?
        if self.use_renode:
            pass
        else:
            raise NotImplementedError("Only renode is supported")
        out = ""
        out += utils.make(
            "renode-test",
            'TEST_FLAGS="--show-log"',
            cwd=self.project_dir,
            env=self.prepare_environment(target=target),
            live=self.print_outputs,
        )
        return out
