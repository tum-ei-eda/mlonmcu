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

from mlonmcu.target.target import Target
from mlonmcu.target.metrics import Metrics

from mlonmcu.logging import get_logger

from .microtvm_zephyr_target import ZephyrMicroTvmPlatformTarget
from .microtvm_arduino_target import ArduinoMicroTvmPlatformTarget
from .microtvm_espidf_target import EspidfMicroTvmPlatformTarget
from .microtvm_host_target import HostMicroTvmPlatformTarget
from .microtvm_etiss_target import EtissMicroTvmPlatformTarget
from .microtvm_spike_target import SpikeMicroTvmPlatformTarget
from .microtvm_gvsoc_target import GVSocMicroTvmPlatformTarget
from .microtvm_corev_ovpsim_target import CoreVOVPSimMicroTvmPlatformTarget
from .microtvm_mlonmcu_target import MlonmcuMicroTvmPlatformTarget

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


register_microtvm_platform_target("microtvm_zephyr", ZephyrMicroTvmPlatformTarget)
register_microtvm_platform_target("microtvm_arduino", ArduinoMicroTvmPlatformTarget)
register_microtvm_platform_target("microtvm_host", HostMicroTvmPlatformTarget)
register_microtvm_platform_target("microtvm_etiss", EtissMicroTvmPlatformTarget)
register_microtvm_platform_target("microtvm_espidf", EspidfMicroTvmPlatformTarget)
register_microtvm_platform_target("microtvm_spike", SpikeMicroTvmPlatformTarget)
register_microtvm_platform_target("microtvm_gvsoc", GVSocMicroTvmPlatformTarget)
register_microtvm_platform_target("microtvm_corev_ovpsim", CoreVOVPSimMicroTvmPlatformTarget)
register_microtvm_platform_target("microtvm_mlonmcu", MlonmcuMicroTvmPlatformTarget)


def create_microtvm_platform_target(name, platform, base=Target):
    class MicroTvmPlatformTarget(base):
        DEFAULTS = {
            **base.DEFAULTS,
            "timeout_sec": 0,  # disabled
        }

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
            if time_s:
                metrics.add("Runtime [s]", time_s)

            if self.platform.profile:
                headers = None
                skip = False
                lines = out.split("\n")
                extracted = []

                def extract_cols(line):
                    x = re.compile(r"(([^\s\[\]]+)(\s\S+)*)(\[.*\])?").findall(line)
                    return [y[0] for y in x]

                for line in lines:
                    if skip:
                        skip = False
                        continue
                    if headers is None:
                        if "Name" in line:
                            headers = extract_cols(line)
                            skip = True
                            continue
                    else:
                        if len(line.strip()) == 0:
                            break
                        cols = extract_cols(line)
                        data = {headers[i]: val for i, val in enumerate(cols)}
                        extracted.append(data)
                        if "Total_time" in line:
                            break
                assert len(extracted) > 0
                metrics = {"default": metrics}
                for item in extracted:
                    if item["Node Name"] == "Total_time":
                        metrics["default"].add("Runtime [s]", float(item["Time(us)"]) / 1e6)
                    else:
                        metrics_ = Metrics()
                        metrics_.add("Runtime [s]", float(item["Time(us)"]) / 1e6)
                        metrics[item["Node Name"]] = metrics_

            return metrics, out, []

        def get_arch(self):
            return "unkwown"

    return MicroTvmPlatformTarget
