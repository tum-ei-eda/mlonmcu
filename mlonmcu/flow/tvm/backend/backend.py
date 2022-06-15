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

from mlonmcu.flow.backend import Backend
from mlonmcu.setup import utils
from mlonmcu.config import str2bool
from mlonmcu.models.model import ModelFormats
from .model_info import get_tflite_model_info, get_relay_model_info
from .tuner import TVMTuner
from .python_utils import prepare_python_environment
from .tvmc_utils import (
    get_target_tvmc_args,
    get_pass_config_tvmc_args,
    get_disabled_pass_tvmc_args,
    get_runtime_executor_tvmc_args,
    get_input_shapes_tvmc_args,
    get_tuning_records_tvmc_args,
)


class TVMBackend(Backend):

    registry = {}

    name = None

    FEATURES = ["autotune", "autotuned", "cmsisnnbyoc", "muriscvnnbyoc", "disable_legalize", "moiopt"]

    DEFAULTS = {
        "print_outputs": False,
        "opt_level": 3,
        "target_device": None,
        "target_mcpu": None,
        "target_march": None,
        "target_model": None,
        "extra_target": None,
        "extra_target_mcpu": None,
        "desired_layout": None,  # optional: NCHW or NHWC
        "disabled_passes": [],  # i.e. AlterOpLayout
        "extra_pass_config": {},  # TODO: some example (fuse_max_depth etc.)
        "use_tuning_results": False,
        "tvmc_extra_args": [],  # Currently compile subcommand only!
        "tvmc_custom_script": None,
        **{("autotuning_" + key): value for key, value in TVMTuner.DEFAULTS.items()},
    }

    REQUIRED = ["tvm.build_dir", "tvm.pythonpath", "tvm.configs_dir"]

    def __init__(self, features=None, config=None, context=None):
        super().__init__(framework="tvm", features=features, config=config, context=context)

        self.model = None  # Actual filename!
        self.model_info = None
        self.input_shapes = None
        self.supported_formats = [ModelFormats.TFLITE, ModelFormats.RELAY]

        self.prefix = "default"
        self.artifacts = (
            []
        )  # TODO: either make sure that ony one model is processed at a time or move the artifacts to the methods
        # TODO: decide if artifacts should be handled by code (str) or file path or binary data
        tuner_config = {  # This would be more compact with a helper function but for now its fine...
            "enable": self.config["autotuning_enable"],
            "results_file": self.config["autotuning_results_file"],
            "append": self.config["autotuning_append"],
            "tuner": self.config["autotuning_tuner"],
            "trials": self.config["autotuning_trials"],
            "early_stopping": self.config["autotuning_early_stopping"],
            "num_workers": self.config["autotuning_num_workers"],
            "max_parallel": self.config["autotuning_max_parallel"],
            "use_rpc": self.config["autotuning_use_rpc"],
            "timeout": self.config["autotuning_timeout"],
            "print_outputs": self.config["print_outputs"],
        }
        self.tuner = TVMTuner(self, config=tuner_config)
        self._tuning_records = None

    @property
    def tuning_records(self):
        if self._tuning_records:
            return self.tuning_records
        elif "autotuning_results_file" in self.config:
            return self.config["autotuning_results_file"]
        else:
            return None

    @tuning_records.setter
    def tuning_records(self, filepath):
        self.tuning_records_file = filepath

    @property
    def pass_config(self):
        base = {"tir.disable_vectorize": True}
        extra = self.config["extra_pass_config"]
        if isinstance(extra, str):
            import ast

            extra = ast.literal_eval(extra)
        assert isinstance(extra, dict)
        base.update(extra)
        return base

    @property
    def target_device(self):
        return self.config["target_device"]

    @property
    def target_mcpu(self):
        return self.config["target_mcpu"]

    @property
    def target_march(self):
        return self.config["target_march"]

    @property
    def target_model(self):
        return self.config["target_model"]

    @property
    def extra_target(self):
        return self.config["extra_target"]

    @property
    def extra_target_mcpu(self):
        return self.config["extra_target_mcpu"]

    @property
    def desired_layout(self):
        return self.config["desired_layout"]

    @property
    def opt_level(self):
        return self.config["opt_level"]

    @property
    def use_tuning_results(self):
        return bool(self.config["use_tuning_results"])

    @property
    def tvmc_extra_args(self):
        return self.config["tvmc_extra_args"]

    @property
    def tvmc_custom_script(self):
        return self.config["tvmc_custom_script"]

    @property
    def disabled_passes(self):
        return self.config["disabled_passes"]

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
    def print_outputs(self):
        return str2bool(self.config["print_outputs"])

    def get_target_details(self):
        ret = {}
        if self.target_device:
            ret["device"] = self.target_device
        if self.target_mcpu:
            ret["mcpu"] = self.target_mcpu
        if self.target_march:
            ret["march"] = self.target_march
        if self.target_model:
            ret["model"] = self.target_model
        return ret

    def get_extra_target_details(self):
        ret = {}
        if self.extra_target_mcpu:
            ret["mcpu"] = self.extra_target_mcpu
        return ret

    def get_tvmc_compile_args(self, out, executor=None, fmt="mlf", target="c", runtime="crt", dump=None):
        assert executor in ["aot", "graph"], "Unsupported TVM executor"
        args = [
            self.model,
            *get_target_tvmc_args(
                target,
                extra_target=self.extra_target,
                target_details=self.get_target_details(),
                extra_target_details=self.get_extra_target_details(),
            ),
            *get_runtime_executor_tvmc_args(runtime, executor),
            *get_pass_config_tvmc_args(self.pass_config),
            *get_disabled_pass_tvmc_args(self.disabled_passes),
            *get_input_shapes_tvmc_args(self.input_shapes),
            *get_tuning_records_tvmc_args(self.use_tuning_results, self.tuning_records),
            *(["--desired-layout", self.desired_layout] if self.desired_layout is not None else []),
            *(["--dump-code", ",".join(dump)] if dump is not None else []),
            *self.tvmc_extra_args,
            *["--opt-level", str(self.opt_level)],
            *["--output", str(out)],
            *["-f", fmt],
            *["--model-format", self.model_format],
        ]
        return args

    def invoke_tvmc(self, command, *args):
        env = prepare_python_environment(self.tvm_pythonpath, self.tvm_build_dir, self.tvm_configs_dir)
        if self.tvmc_custom_script is None:
            pre = ["-m", "tvm.driver.tvmc"]
        else:
            pre = [self.tvmc_custom_script]
        return utils.python(*pre, command, *args, live=self.print_outputs, print_output=False, env=env)

    def invoke_tvmc_compile(self, out, dump=None):
        args = self.get_tvmc_compile_args(out)
        return self.invoke_tvmc("compile", *args)

    def load_model(self, model):
        self.model = model
        # TODO: path model class instead of path!
        # fmt = self.model.formats[0]
        ext = os.path.splitext(model)[1][1:]
        fmt = ModelFormats.from_extension(ext)
        if fmt == ModelFormats.TFLITE:
            self.model_format = "tflite"
            with open(model, "rb") as handle:
                model_buf = handle.read()
                self.model_info = get_tflite_model_info(model_buf)
        elif fmt == ModelFormats.RELAY:
            # Warning: the wrapper generateion does currently not work because of the
            # missing possibility to get the relay models input names and shapes
            self.model_format = "relay"
            with open(model, "r") as handle:
                mod_text = handle.read()
            self.model_info = get_relay_model_info(mod_text)
        else:
            raise RuntimeError(f"Unsupported model format '{fmt.name}' for backend '{self.name}'")
        self.input_shapes = {tensor.name: tensor.shape for tensor in self.model_info.in_tensors}
