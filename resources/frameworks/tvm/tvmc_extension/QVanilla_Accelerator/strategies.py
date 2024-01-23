# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Strategies for the q_vanilla_accelerator accelerator"""


from tvm import relay
from tvm.relay import op as _op
from tvm.relay.qnn.strategy.generic import wrap_topi_schedule, wrap_topi_qnn_conv2d
from tvm import topi


@relay.op.strategy.override_native_generic_func("qnn_conv2d_strategy")
def qnn_conv2d_strategy(attrs, inputs, out_type, target):
    print("qnn strategy")
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_topi_qnn_conv2d(topi.hexagon.qnn_conv2d), wrap_topi_schedule(topi.hexagon.schedule_qnn_conv2d)
    )

    return strategy
