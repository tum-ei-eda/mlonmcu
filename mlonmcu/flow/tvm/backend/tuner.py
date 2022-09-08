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


class TVMTuner:
    DEFAULTS = {
        "enable": False,
        "print_outputs": False,
        "results_file": None,
        "append": None,
        "tuner": "ga",  # Options: ga,gridsearch,random,xgb,xgb_knob,xgb-rank
        "trials": 10,  # TODO: increase to 100?
        "early_stopping": None,  # calculate default dynamically
        "num_workers": 1,
        "max_parallel": 1,
        "use_rpc": False,
        "timeout": 100,
        "mode": "autotvm",  # Options: autotvm, auto_scheduler
        "visualize": False,
        "tasks": None,
    }

    def __init__(self, backend, config=None):
        self.backend = backend
        self.config = config if config is not None else {}
