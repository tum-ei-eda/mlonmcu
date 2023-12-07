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
import multiprocessing

from mlonmcu.flow.backend import Backend
from mlonmcu.setup import utils
from mlonmcu.config import str2bool
from mlonmcu.logging import get_logger
from .model_info import get_model_info, get_fallback_model_info, get_supported_formats, get_model_format
from .python_utils import prepare_python_environment
from .tvmc_utils import (
    get_target_tvmc_args,
    get_pass_config_tvmc_args,
    get_disabled_pass_tvmc_args,
    get_runtime_executor_tvmc_args,
    get_input_shapes_tvmc_args,
    get_tuning_records_tvmc_args,
)

logger = get_logger()


class TVMBackend(Backend):
    registry = {}

    name = None

    FEATURES = ["autotuned", "cmsisnnbyoc", "muriscvnnbyoc", "disable_legalize", "moiopt"]

    DEFAULTS = {
        "print_outputs": False,
        "opt_level": 3,
        "target_device": None,
        "target_mcpu": None,
        "target_march": None,
        "target_model": None,
        "target_mtriple": None,
        "target_mabi": None,
        "target_mattr": None,
        "extra_target": None,
        "extra_target_mcpu": None,
        "desired_layout": None,  # optional: NCHW or NHWC
        "disabled_passes": [],  # i.e. AlterOpLayout
        "extra_pass_config": {},  # TODO: some example (fuse_max_depth etc.)
        "use_tuning_results": False,
        "tvmc_extra_args": [],  # Currently compile subcommand only!
        "tvmc_custom_script": None,
        # See https://github.com/apache/tvm/blob/1115fd9bc261619ffa0539746ae0aebc46232dc6/python/tvm/autotvm/tophub.py
        "tophub_url": None,
        "num_threads": multiprocessing.cpu_count(),
    }

    REQUIRED = []

    OPTIONAL = ["tvm.build_dir", "tvm.pythonpath", "tvm.configs_dir", "tvm.use_tlcpack"]

    def __init__(self, target="c", executor=None, runtime="crt", fmt="mlf", features=None, config=None):
        super().__init__(framework="tvm", features=features, config=config)

        self.model = None  # Actual filename!
        self.model_info = None
        self.input_shapes = None
        self.supported_formats = get_supported_formats()
        self.target = target
        self.runtime = runtime
        self.executor = executor
        self.fmt = fmt

        self.prefix = "default"
        self.artifacts = (
            []
        )  # TODO: either make sure that ony one model is processed at a time or move the artifacts to the methods
        self._tuning_records = None

    @property
    def tuning_records(self):
        if self._tuning_records:
            return self._tuning_records
        elif "autotuning_results_file" in self.config and self.config["autotuning_results_file"]:
            return self.config["autotuning_results_file"]
        else:
            return None

    @tuning_records.setter
    def tuning_records(self, filepath):
        self._tuning_records = filepath

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
    def target_mtriple(self):
        return self.config["target_mtriple"]

    @property
    def target_mabi(self):
        return self.config["target_mabi"]

    @property
    def target_mattr(self):
        return self.config["target_mattr"]

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
        value = self.config["use_tuning_results"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

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
    def tophub_url(self):
        return self.config["tophub_url"]

    @property
    def print_outputs(self):
        value = self.config["print_outputs"]
        return str2bool(value) if not isinstance(value, (bool, int)) else value

    @property
    def use_tlcpack(self):
        value = self.config["tvm.use_tlcpack"]
        return str2bool(value, allow_none=True) if not isinstance(value, (bool, int)) else value

    def num_threads(self):
        return self.config["num_threads"]

    def get_target_details(self):
        ret = {}
        if self.target_device:
            ret["device"] = self.target_device
        if self.target_mcpu:
            ret["mcpu"] = self.target_mcpu
        if self.target_march:
            ret["march"] = self.target_march
        if self.target_mtriple:
            ret["mtriple"] = self.target_mtriple
        if self.target_mabi:
            ret["mabi"] = self.target_mabi
        if self.target_mattr:
            ret["mattr"] = self.target_mattr
        if self.target_model:
            ret["model"] = self.target_model
        return ret

    def get_extra_target_details(self):
        ret = {}
        if self.extra_target_mcpu:
            ret["mcpu"] = self.extra_target_mcpu
        return ret

    def get_tvmc_compile_args(self, out, dump=None):
        assert self.executor is not None
        assert self.executor in ["aot", "graph"], "Unsupported TVM executor"
        args = [
            self.model,
            *get_target_tvmc_args(
                self.target,
                extra_target=self.extra_target,
                target_details=self.get_target_details(),
                extra_target_details=self.get_extra_target_details(),
            ),
            *get_runtime_executor_tvmc_args(self.runtime, self.executor),
            *get_pass_config_tvmc_args(self.pass_config),
            *get_disabled_pass_tvmc_args(self.disabled_passes),
            *get_input_shapes_tvmc_args(self.input_shapes),
            *get_tuning_records_tvmc_args(self.use_tuning_results, self.tuning_records),
            *(["--desired-layout", self.desired_layout] if self.desired_layout is not None else []),
            *(["--dump-code", ",".join(dump)] if dump is not None else []),
            *self.tvmc_extra_args,
            *["--opt-level", str(self.opt_level)],
            *["--output", str(out)],
            *["-f", self.fmt],
            *["--model-format", self.model_format],
        ]
        return args

    def invoke_tvmc(self, command, *args, cwd=None):
        env = prepare_python_environment(
            None if self.use_tlcpack else self.tvm_pythonpath,
            None if self.use_tlcpack else self.tvm_build_dir,
            None if self.use_tlcpack else self.tvm_configs_dir,
            tophub_url=self.tophub_url,
            num_threads=self.num_threads,
        )
        if self.use_tlcpack:
            pre = ["tvmc"]
            return utils.exec_getout(
                *pre, command, *args, live=self.print_outputs, print_output=False, env=env, cwd=cwd
            )
        else:
            if self.tvmc_custom_script is None:
                pre = ["-m", "tvm.driver.tvmc"]
            else:
                pre = [self.tvmc_custom_script]
            return utils.python(*pre, command, *args, live=self.print_outputs, print_output=False, env=env, cwd=cwd)

    def invoke_tvmc_compile(self, out, dump=None, cwd=None):
        args = self.get_tvmc_compile_args(out, dump=dump)
        return self.invoke_tvmc("compile", *args, cwd=cwd)

    def load_model(self, model, input_shapes=None, output_shapes=None, input_types=None, output_types=None):
        self.model = model
        # TODO: path model class instead of path!
        # fmt = self.model.formats[0]
        need_model_info = True
        if input_shapes:
            self.input_shapes = input_shapes
            if output_shapes and input_types and output_types:
                need_model_info = False
                self.model_format, self.model_info = get_fallback_model_info(
                    model, input_shapes, output_shapes, input_types, output_types, backend_name=self.name
                )
        if need_model_info:
            try:
                self.model_format, self.model_info = get_model_info(model, backend_name=self.name)
            except Exception as e:
                logger.warning(
                    "Fetching of Model Info failed (%s). Falling back to Relay-based info.", type(e).__name__
                )
                self.model_format = get_model_format(model)
                self.model_info = None

            if self.model_info and not self.input_shapes:
                self.input_shapes = {tensor.name: tensor.shape for tensor in self.model_info.in_tensors}
            self.model_info = None
