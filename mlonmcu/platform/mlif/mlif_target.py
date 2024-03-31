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
import os
from math import ceil
from pathlib import Path
from enum import IntEnum

from mlonmcu.target import get_targets, Target
from mlonmcu.artifact import Artifact, ArtifactFormat
from mlonmcu.logging import get_logger

from .interfaces import ModelSupport

logger = get_logger()


def get_mlif_platform_targets():
    return get_targets()


class MlifExitCode(IntEnum):
    ERROR = 0x10
    INVALID_SIZE = 0x11
    OUTPUT_MISSMATCH = 0x12

    @classmethod
    def values(cls):
        return list(map(int, cls))


def create_mlif_platform_target(name, platform, base=Target):
    class MlifPlatformTarget(base):
        DEFAULTS = {
            **base.DEFAULTS,
        }

        def __init__(self, features=None, config=None):
            super().__init__(name=name, features=features, config=config)
            self.platform = platform
            self.validation_result = None

        def exec(self, program, *args, cwd=os.getcwd(), **kwargs):
            ins_file = None
            num_inputs = 0
            in_interface = None
            out_interface = None
            batch_size = 1
            encoding = "utf-8"
            model_info_file = self.platform.model_info_file  # TODO: replace workaround (add model info to platform?)
            if self.platform.set_inputs or self.platform.get_outputs:
                # first figure out how many inputs are provided
                if model_info_file is not None:
                    import yaml

                    with open(model_info_file, "r") as f:
                        model_info_data = yaml.safe_load(f)
                else:
                    model_info_data = None
                if self.platform.inputs_artifact is not None:
                    import numpy as np

                    data = np.load(self.platform.inputs_artifact, allow_pickle=True)
                    num_inputs = len(data)
                else:
                    data = None
                model_support = ModelSupport(
                    in_interface=self.platform.set_inputs_interface,
                    out_interface=self.platform.get_outputs_interface,
                    model_info=model_info_data,
                    target=self,
                    batch_size=self.platform.batch_size,
                    inputs_data=data,
                )
                in_interface = model_support.in_interface
                out_interface = model_support.out_interface
                batch_size = model_support.batch_size
                if out_interface == "stdout_raw":
                    encoding = None
            outs_file = None
            ret = ""
            artifacts = []
            num_batches = max(ceil(num_inputs / batch_size), 1)
            processed_inputs = 0
            # remaining_inputs = num_inputs
            outs_data = []
            stdin_data = None
            for idx in range(num_batches):
                # print("idx", idx)
                # current_batch_size = max(min(batch_size, remaining_inputs), 1)
                if processed_inputs < num_inputs:
                    if in_interface == "filesystem":
                        batch_data = data[idx * batch_size : ((idx + 1) * batch_size)]
                        # print("batch_data", batch_data, type(batch_data))
                        ins_file = Path(cwd) / "ins.npy"
                        np.save(ins_file, batch_data)
                    elif in_interface == "stdin":
                        raise NotImplementedError
                    elif in_interface == "stdin_raw":
                        batch_data = data[idx * batch_size : ((idx + 1) * batch_size)]
                        # print("batch_data", batch_data, type(batch_data))
                        stdin_data = b""
                        for cur_data in batch_data:
                            # print("cur_data", cur_data)
                            for key, value in cur_data.items():
                                # print("key", key)
                                # print("value", value, type(value))
                                # print("value.tostring", value.tostring())
                                stdin_data += value.tostring()
                        # TODO: check that stdin_data has expected size
                        # This is just a placeholder example!
                        # stdin_data = "input[0] = {0, 1, 2, ...};\nDONE\n""".encode()
                        # stdin_data *= 200
                        # raise NotImplementedError
                        # TODO: generate input stream here!

                ret_, artifacts_ = super().exec(
                    program, *args, cwd=cwd, **kwargs, stdin_data=stdin_data, encoding=encoding
                )
                if self.platform.get_outputs:
                    if out_interface == "filesystem":
                        import numpy as np

                        outs_file = Path(cwd) / "outs.npy"
                        with np.load(outs_file) as out_data:
                            outs_data.extend(dict(out_data))
                    elif out_interface == "stdout":
                        # TODO: get output_data from stdout
                        raise NotImplementedError
                    elif out_interface == "stdout_raw":
                        # DUMMY BELOW
                        assert model_info_data is not None
                        # print("model_info_data", model_info_data)
                        # dtype = "int8"
                        # shape = [1, 10]
                        # input("!")
                        # print("ret_", ret_, type(ret_))
                        # out_idx = 0
                        x = ret_  # Does this copy?
                        while True:
                            out_data_temp = {}
                            # substr = ret_[ret_.find("-?-".encode())+3:ret_.find("-!-".encode())]
                            # print("substr", substr, len(substr))
                            found_start = x.find("-?-".encode())
                            # print("found_start", found_start)
                            if found_start < 0:
                                break
                            x = x[found_start + 3 :]
                            # print("x[:20]", x[:20])
                            found_end = x.find("-!-".encode())
                            # print("found_end", found_end)
                            assert found_end >= 0
                            x_ = x[:found_end]
                            x = x[found_end + 3 :]
                            # print("x[:20]", x[:20])
                            # print("x_", x_)
                            # out_idx += 1
                            dtype = model_info_data["output_types"][0]
                            arr = np.frombuffer(x_, dtype=dtype)
                            # print("arr", arr)
                            shape = model_info_data["output_shapes"][0]
                            arr = arr.reshape(shape)
                            # print("arr2", arr)
                            assert len(model_info_data["output_names"]) == 1, "Multi-output models not yet supported"
                            out_name = model_info_data["output_names"][0]
                            out_data_temp[out_name] = arr
                            outs_data.append(out_data_temp)
                        # {"output_0": arr}])
                        ret_ = ret_.decode("utf-8", errors="replace")
                        # raise NotImplementedError
                    else:
                        assert False
                ret += ret_
                artifacts += artifacts_
            # print("outs_data", outs_data)
            # input("$")
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

        def get_metrics(self, elf, directory, handle_exit=None):
            # This is wrapper around the original exec function to catch special return codes thrown by the inout data
            # feature (TODO: catch edge cases: no input data available (skipped) and no return code (real hardware))
            if self.platform.validate_outputs or not self.platform.skip_check:

                def _handle_exit(code, out=None):
                    if handle_exit is not None:
                        code = handle_exit(code, out=out)
                    if code == 0:
                        self.validation_result = True
                    else:
                        if code in MlifExitCode.values():
                            reason = MlifExitCode(code).name
                            logger.error("A platform error occured during the simulation. Reason: %s", reason)
                            if code == MlifExitCode.OUTPUT_MISSMATCH:
                                self.validation_result = False
                                if not self.platform.fail_on_error:
                                    code = 0
                    return code

            else:
                _handle_exit = handle_exit

            metrics, out, artifacts = super().get_metrics(elf, directory, handle_exit=_handle_exit)

            if self.platform.validate_outputs or not self.platform.skip_check:
                metrics.add("Validation", self.validation_result)
            return metrics, out, artifacts

        def get_platform_defs(self, platform):
            ret = super().get_platform_defs(platform)
            target_system = self.get_target_system()
            ret["TARGET_SYSTEM"] = target_system
            return ret

    return MlifPlatformTarget
