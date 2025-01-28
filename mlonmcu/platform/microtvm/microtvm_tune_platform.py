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
"""MicroTVM Tune Platform"""
from mlonmcu.config import str2bool
from .microtvm_target_platform import MicroTvmTargetPlatform
from ..tvm.tvm_tune_platform import TvmTunePlatform


class MicroTvmTunePlatform(TvmTunePlatform, MicroTvmTargetPlatform):
    """MicroTVM Tune platform class."""

    FEATURES = TvmTunePlatform.FEATURES | MicroTvmTargetPlatform.FEATURES

    DEFAULTS = {
        **TvmTunePlatform.DEFAULTS,
        **MicroTvmTargetPlatform.DEFAULTS,
    }

    REQUIRED = TvmTunePlatform.REQUIRED | MicroTvmTargetPlatform.REQUIRED

    @property
    def experimental_tvmc_micro_tune(self):
        value = self.config["experimental_tvmc_micro_tune"]
        return str2bool(value)

    def invoke_tvmc_micro_tune(self, *args, target=None, list_options=False, **kwargs):
        all_args = []
        all_args.extend(args)
        template_args = self.get_template_args(target)
        all_args.extend(template_args)
        return self.invoke_tvmc_micro("tune", *all_args, target=target, list_options=list_options, **kwargs)

    def invoke_tvmc_tune(self, *args, target=None, **kwargs):
        return self.invoke_tvmc_micro_tune(*args, target=target, **kwargs)

    def _tune_model(self, model_path, backend, target):
        assert self.experimental_tvmc_micro_tune, "Microtvm tuning requires experimental_tvmc_micro_tune"

        return super()._tune_model(model_path, backend, target)
