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
from mlonmcu.artifact import Artifact, ArtifactFormat

from mlonmcu.logging import get_logger

logger = get_logger()


def name2device(name):
    return name.replace("tvm_", "")


def create_tvm_platform_target(name, platform, base=Target):
    class TvmPlatformTarget(base):
        DEFAULTS = {
            **base.DEFAULTS,
            "timeout_sec": 0,  # disabled
        }

        def __init__(self, features=None, config=None):
            super().__init__(name=name, features=features, config=config)
            self.platform = platform
            self.device = name2device(name)

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

            ins_file = None
            num_inputs = 0
            batch_size = 1
            if self.platform.set_inputs:
                interface = self.platform.set_inputs_interface
                if interface == "auto":
                    if self.supports_filesystem:
                        interface = "filesystem"
                else:
                    assert interface in ["filesystem"]
                if interface == "filesystem":
                    ins_file = self.platform.ins_file
                    if ins_file is None:
                        assert self.platform.inputs_artifact is not None
                        import numpy as np

                        data = np.load(self.platform.inputs_artifact, allow_pickle=True)
                        num_inputs = len(data)
                        ins_file = Path(cwd) / "ins.npz"
            outs_file = None
            print_top = self.platform.print_top
            if self.platform.get_outputs:
                interface = self.platform.get_outputs_interface
                if interface == "auto":
                    if self.supports_filesystem:
                        interface = "filesystem"
                    elif self.supports_stdout:
                        interface = "stdout"
                else:
                    assert interface in ["filesystem", "stdout"]
                if interface == "filesystem":
                    outs_file = self.platform.outs_file
                    if outs_file is None:
                        outs_file = Path(cwd) / "outs.npz"
                elif interface == "stdout":
                    print_top = 1e6

            ret = ""
            artifacts = []
            num_batches = max(round(num_inputs / batch_size), 1)
            processed_inputs = 0
            remaining_inputs = num_inputs
            outs_data = []
            for idx in range(num_batches):
                current_batch_size = max(min(batch_size, remaining_inputs), 1)
                assert current_batch_size == 1
                if processed_inputs < num_inputs:
                    in_data = data[idx]
                    np.savez(ins_file, **in_data)
                    processed_inputs += 1
                    remaining_inputs -= 1
                else:
                    ins_file = None
                ret_, artifacts_ = self.platform.run(
                    program, self, cwd=cwd, ins_file=ins_file, outs_file=outs_file, print_top=print_top
                )
                ret += ret_
                if self.platform.get_outputs:
                    interface = self.platform.get_outputs_interface
                    if interface == "auto":
                        if self.supports_filesystem:
                            interface = "filesystem"
                        elif self.supports_stdout:
                            interface = "stdout"
                    else:
                        assert interface in ["filesystem", "stdout"]
                    if interface == "filesystem":
                        import numpy as np

                        with np.load(outs_file) as out_data:
                            outs_data.append(dict(out_data))
                    elif interface == "stdout":
                        raise NotImplementedError
                    else:
                        assert False
            if len(outs_data) > 0:
                outs_path = Path(cwd) / "outputs.npy"
                np.save(outs_path, outs_data)
                with open(outs_path, "rb") as f:
                    outs_raw = f.read()
                outputs_artifact = Artifact(
                    "outputs.npy", raw=outs_raw, fmt=ArtifactFormat.BIN, flags=("outputs", "npy")
                )
                artifacts.append(outputs_artifact)

            return ret, artifacts

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
            artifacts = []
            if self.print_outputs:
                out, artifacts = self.exec(elf, cwd=directory, live=True, handle_exit=handle_exit)
            else:
                out, artifacts = self.exec(
                    elf,
                    cwd=directory,
                    live=False,
                    print_func=lambda *args, **kwargs: None,
                    handle_exit=handle_exit,
                )
            mean_ms, _, max_ms, min_ms, _ = self.parse_stdout(out)

            metrics = Metrics()
            mean_s = mean_ms / 1e3 if mean_ms is not None else mean_ms
            min_s = min_ms / 1e3 if min_ms is not None else min_ms
            max_s = max_ms / 1e3 if max_ms is not None else max_ms
            if (
                self.platform.number == 1
                and self.platform.repeat == 1
                and (not self.platform.total_time or self.platform.aggregate != "none")
            ):
                metrics.add("Runtime [s]", mean_s)
            else:
                if self.platform.total_time:
                    metrics.add("Total Runtime [s]", mean_s * self.platform.number)
                if self.platform.aggregate == "all":
                    metrics.add("Average Runtime [s]", mean_s)
                    metrics.add("Min Runtime [s]", min_s)
                    metrics.add("Max Runtime [s]", max_s)
                elif self.platform.aggregate in ["avg", "mean"]:
                    metrics.add("Average Runtime [s]", mean_s)
                elif self.platform.aggregate == "min":
                    metrics.add("Min Runtime [s]", min_s)
                elif self.platform.aggregate == "max":
                    metrics.add("Max Runtime [s]", max_s)

            if self.platform.profile:
                headers = None
                lines = out.split("\n")
                extracted = []

                def extract_cols(line):
                    x = re.compile(r"(([^\s\[\]]+)(\s\S+)*)(\[.*\])?").findall(line)
                    return [y[0] for y in x]

                for line in lines:
                    if "---" in line:
                        break
                    if headers is None:
                        if "Name" in line:
                            headers = extract_cols(line)
                    else:
                        cols = extract_cols(line)
                        data = {headers[i]: val for i, val in enumerate(cols)}
                        extracted.append(data)
                assert len(extracted) > 0
                metrics = {"default": metrics}
                for item in extracted:
                    metrics_ = Metrics()
                    metrics_.add("Runtime [s]", float(item["Duration (us)"].replace(",", "")) / 1e6)
                    metrics[item["Name"]] = metrics_

            return metrics, out, artifacts

        def get_arch(self):
            return "unkwown"

        def update_environment(self, env):
            # TODO: implement in base class?
            pass

        @property
        def supports_filesystem(self):
            return True

        @property
        def supports_stdout(self):
            return True

    return TvmPlatformTarget
