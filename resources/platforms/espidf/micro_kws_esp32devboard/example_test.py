#!/usr/bin/env python
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

from __future__ import division, print_function, unicode_literals

import ttfw_idf


@ttfw_idf.idf_example_test(env_tag="Example_GENERIC", target=["esp32", "esp32s2", "esp32c3"], ci_target=["esp32"])
def test_examples_hello_world(env, extra_data):
    app_name = "hello_world"
    dut = env.get_dut(app_name, "examples/get-started/hello_world")
    dut.start_app()
    res = dut.expect(ttfw_idf.MINIMUM_FREE_HEAP_SIZE_RE)
    if not res:
        raise ValueError("Maximum heap size info not found")
    ttfw_idf.print_heap_size(app_name, dut.app.config_name, dut.TARGET, res[0])


if __name__ == "__main__":
    test_examples_hello_world()
