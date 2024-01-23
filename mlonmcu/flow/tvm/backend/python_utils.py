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


def prepare_python_environment(
    pythonpath, tvm_build_dir, tvm_configs_dir, tophub_url=None, num_threads=None, debug_cfg=None
):
    env = os.environ.copy()
    if debug_cfg:
        env["TVM_LOG_DEBUG"] = debug_cfg
    if pythonpath:
        env["PYTHONPATH"] = str(pythonpath)
    if tvm_build_dir:
        env["TVM_LIBRARY_PATH"] = str(tvm_build_dir)
    if tvm_configs_dir:
        env["TVM_CONFIGS_JSON_DIR"] = str(tvm_configs_dir)
    if tophub_url:
        env["TOPHUB_LOCATION"] = tophub_url
    if num_threads:
        env["TVM_NUM_THREADS"] = str(num_threads)
    # Use all cores/threads for building models and let OS scheduler decide on the mapping
    env["TVM_BIND_THREADS"] = str(0)
    return env
