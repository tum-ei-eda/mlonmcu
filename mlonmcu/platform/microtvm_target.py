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
import re
import os
from pathlib import Path

from mlonmcu.target.target import Target
from mlonmcu.target.metrics import Metrics

from mlonmcu.logging import get_logger

logger = get_logger()


def name2template(name):
    return name.replace("microtvm_", "")


MICROTVM_PLATFORM_TARGET_REGISTRY = {}


def register_microtvm_platform_target(target_name, t, override=False):
    global MICROTVM_PLATFORM_TARGET_REGISTRY

    if target_name in MICROTVM_PLATFORM_TARGET_REGISTRY and not override:
        raise RuntimeError(f"MicroTVM platform target {target_name} is already registered")
    MICROTVM_PLATFORM_TARGET_REGISTRY[target_name] = t


def get_microtvm_platform_targets():
    return MICROTVM_PLATFORM_TARGET_REGISTRY


class TemplateMicroTvmPlatformTarget(Target):

    FEATURES = Target.FEATURES + []

    DEFAULTS = {
        **Target.DEFAULTS,
    }
    REQUIRED = Target.REQUIRED + ["tvm.build_dir"]

    def __init__(self, name=None, features=None, config=None):
        super().__init__(name=name, features=features, config=config)
        self.template_path = None
        self.option_names = []
        # self.platform = platform
        # self.template = name2template(name)

    def get_project_options(self):
        return {key: value for key, value in self.config if key in self.option_names}


# class ArduinoMicroTvmPlatformTarget(TemplateMicroTvmPlatformTarget):
#
#     FEATURES = Target.FEATURES + []
#
#     DEFAULTS = {
#         **Target.DEFAULTS,
#         "project_type": "?",
#         "warning_as_error": False,
#         "arduino_board": "?",
#         # "arduino_clicmd?": "?",
#         "verbose": False,
#         "port": -1,
#     }
#     REQUIRED = Target.REQUIRED + ["arduino.install_dir"]
#
#     def __init__(self, name=None, features=None, config=None):
#         super().__init__(name=name, features=features, config=config)
#         self.template_path = None
#         self.option_names = ["project_type", "warning_as_error", "arduino_board", "verbose", "port"]
#         # self.platform = platform
#         # self.template = name2template(name)
#
#     @property
#     def arduino_install_dir(self):
#         return Path(self.config["arduino.install_dir"])
#
#     def get_project_options(self):
#         ret = super().get_project_options()
#         ret.update({"arduino_clicmd": ?})
#
#
# class ZephyrMicroTvmPlatformTarget(TemplateMicroTvmPlatformTarget):
#
#     FEATURES = Target.FEATURES + []
#
#     DEFAULTS = {
#         **Target.DEFAULTS,
#         "extra_files_tar": None,
#         "project_type": "?",
#         "zephyr_board": "?",
#         # "zephyr_base": "?",
#         # "west_cmd": "?",
#         "verbose": False,
#         "warning_as_error": True,
#         "compile_definitions": "",
#         "config_main_stack_size": -1,
#         "gdbserver_port": -1,
#         "nrfjprog_snr": None,
#         "openocd_serial": None,
#     }
#     REQUIRED = Target.REQUIRED + ["zephyr.install_dir"]
#
#     def __init__(self, name=None, features=None, config=None):
#         super().__init__(name=name, features=features, config=config)
#         self.template_path = None
#         self.option_names = ["extra_files_tar", "project_type", "zephyr_board", "verbose", "warning_as_error", "compile_definitions", "config_main_stack_size", "gdbserver_port", "nrfjprog_snr", "openocd_serial"]
#         # self.platform = platform
#         # self.template = name2template(name)
#
#     @property
#     def zephyr_install_dir(self):
#         return Path(self.config["zephyr.install_dir"])
#
#     def get_project_options(self):
#         ret = super().get_project_options()
#         ret.update({"zephyr_base": self.zephyr_install_dir})


class HostMicroTvmPlatformTarget(TemplateMicroTvmPlatformTarget):

    FEATURES = Target.FEATURES + []

    DEFAULTS = {
        **Target.DEFAULTS,
        "verbose": False,
    }
    REQUIRED = Target.REQUIRED + ["tvm.build_dir"]

    def __init__(self, name=None, features=None, config=None):
        super().__init__(name=name, features=features, config=config)
        self.template_path = self.tvm_build_dir / "standalone_crt" / "template" / "host"
        self.option_names = ["verbose"]

    @property
    def tvm_build_dir(self):
        return Path(self.config["tvm.build_dir"])


class EtissvpMicroTvmPlatformTarget(TemplateMicroTvmPlatformTarget):

    FEATURES = Target.FEATURES + []

    DEFAULTS = {
        **Target.DEFAULTS,
        "extra_files_tar": None,
        "project_type": "?",
        "verbose": False,
        "warning_as_error": True,
        "compile_definitions": "",
        "config_main_stack_size": -1,
        # "riscv_path": "?",
        # "etiss_path": "?",
        # "etissvp_script": "?",
        "etissvp_script_args": "?",
        "transport": True,
    }
    REQUIRED = Target.REQUIRED + ["microtvm_etissvp.src_dir", "riscv_gcc.install_dir", "etiss.install_dir"]

    def __init__(self, name=None, features=None, config=None):
        super().__init__(name=name, features=features, config=config)
        self.template_path = self.tvm_build_dir / "template_project"
        self.option_names = [
            "extra_files_tar",
            "project_type",
            "verbose",
            "warning_as_error",
            "compile_definitions",
            "config_main_stack_size",
            "etissvp_script_args",
            "transport",
        ]

    @property
    def microtvm_etissvp_src_dir(self):
        return Path(self.config["microtvm_etissvp.src_dir"])

    @property
    def riscv_gcc_install_dir(self):
        return Path(self.config["riscv_gcc_install_dir"])

    @property
    def etiss_install_dir(self):
        return Path(self.config["etiss.install_dir"])

    def get_project_options(self):
        ret = super().get_project_options()
        ret.update(
            {
                "riscv_path": self.riscv_gcc_install_dir,
                "etiss_path": self.etiss_install_dir,
                "etissvp_script": self.etiss_install_dir / "bin" / "run_helper.sh",
            }
        )
        return ret


# class EspidfMicroTvmPlatformTarget(TemplateMicroTvmPlatformTarget):
#
#     FEATURES = Target.FEATURES + []
#
#     DEFAULTS = {
#         **Target.DEFAULTS,
#         "verbose": False,
#         "?": "?",  # TODO
#     }
#     REQUIRED = Target.REQUIRED + ["microtvm_espidf.src_dir", "espidf.src_dir", "espidf.install_dir"]
#
#     def __init__(self, name=None, features=None, config=None):
#         super().__init__(name=name, features=features, config=config)
#         self.template_path = None

# class SpikeMicroTvmPlatformTarget(TemplateMicroTvmPlatformTarget):
#
#     FEATURES = Target.FEATURES + []
#
#     DEFAULTS = {
#         **Target.DEFAULTS,
#         "verbose": False,
#         "?": "?",  # TODO
#     }
#     REQUIRED = Target.REQUIRED + ["microtvm_espidf.src_dir", "espidf.src_dir", "espidf.install_dir"]
#
#     def __init__(self, name=None, features=None, config=None):
#         super().__init__(name=name, features=features, config=config)
#         self.template_path = None


# register_microtvm_platform_target("microtvm_template", ZephyrMicroTvmPlatformTarget)
# register_microtvm_platform_target("microtvm_zephyr", ZephyrMicroTvmPlatformTarget)
# register_microtvm_platform_target("microtvm_arduino", ArduinoMicroTvmPlatformTarget)
register_microtvm_platform_target("microtvm_host", HostMicroTvmPlatformTarget)
register_microtvm_platform_target("microtvm_etissvp", EtissvpMicroTvmPlatformTarget)
# register_microtvm_platform_target("microtvm_espidf", EtissvpMicroTvmPlatformTarget)
# register_microtvm_platform_target("microtvm_spike", SpikeMicroTvmPlatformTarget)


def create_microtvm_platform_target(name, platform, base=Target):
    class MicroTvmPlatformTarget(base):

        FEATURES = base.FEATURES + []

        DEFAULTS = {
            **base.DEFAULTS,
            "timeout_sec": 0,  # disabled
        }
        REQUIRED = base.REQUIRED + []

        def __init__(self, features=None, config=None):
            super().__init__(name=name, features=features, config=config)
            self.platform = platform
            self.template = name2template(name)

        @property
        def timeout_sec(self):
            return int(self.config["timeout_sec"])

        @property
        def repeat(self):
            return None  # This is handled at the platform level

        def exec(self, program, *args, cwd=os.getcwd(), **kwargs):
            """Use target to execute a executable with given arguments"""
            if len(args) > 0:
                raise RuntimeError("Program arguments are not supported for real hardware devices")

            assert self.platform is not None, "TVM targets need a platform to execute programs"

            if self.timeout_sec > 0:
                raise NotImplementedError

            ret = self.platform.run(program, self)
            return ret

        def parse_stdout(self, out):
            mean_ms = None
            median_ms = None
            max_ms = None
            min_ms = None
            std_ms = None
            found = False
            for line in out.split("\n"):
                if found:
                    match = re.compile(r"\s+(\d*\.\d+)\s+(\d*\.\d+)\s+(\d*\.\d+)\s+(\d*\.\d+)\s+(\d*\.\d+)").findall(
                        line
                    )
                    assert len(match) == 1
                    groups = match[0]
                    mean_ms, median_ms, max_ms, min_ms, std_ms = (
                        float(groups[0]),
                        float(groups[1]),
                        float(groups[2]),
                        float(groups[3]),
                        float(groups[4]),
                    )
                    break
                if re.compile(r"\s+mean \(ms\)\s+median \(ms\)\s+max \(ms\)\s+min \(ms\)\s+std \(ms\)").match(line):
                    found = True
            return mean_ms, median_ms, max_ms, min_ms, std_ms

        def get_metrics(self, elf, directory, handle_exit=None):
            if self.print_outputs:
                out = self.exec(elf, cwd=directory, live=True, handle_exit=handle_exit)
            else:
                out = self.exec(
                    elf,
                    cwd=directory,
                    live=False,
                    print_func=lambda *args, **kwargs: None,
                    handle_exit=handle_exit,
                )
            mean_ms, _, _, _, _ = self.parse_stdout(out)

            metrics = Metrics()
            time_s = mean_ms / 1e3 if mean_ms is not None else mean_ms
            metrics.add("Mean Runtime [s]", time_s)

            return metrics, out, []

        def get_arch(self):
            return "unkwown"

    return MicroTvmPlatformTarget
