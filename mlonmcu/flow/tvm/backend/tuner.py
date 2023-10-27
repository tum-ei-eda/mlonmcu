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


def get_autotuning_defaults():
    return {
        "mode": None,
        "results_file": None,
        "append": None,
        "trials": 10,  # TODO: increase to 100?
        "trials_single": None,
        "early_stopping": None,  # calculate default dynamically
        "num_workers": None,
        "max_parallel": 1,
        "use_rpc": False,
        "timeout": 100,
        "tasks": None,
        "visualize": False,
        "visualize_file": None,
        "visualize_live": False,
    }


def get_autotvm_defaults():
    return {
        "enable": False,
        "tuner": "ga",  # Options: ga,gridsearch,random,xgb,xgb_knob,xgb-rank
    }


def get_autoscheduler_defaults():
    return {
        "enable": False,
        "include_simple_tasks": False,
        "log_estimated_latency": True,
    }


def get_metascheduler_defaults():
    return {
        "enable": False,
    }
