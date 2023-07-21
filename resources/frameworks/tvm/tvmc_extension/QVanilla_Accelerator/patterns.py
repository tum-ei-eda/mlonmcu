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
"""Relay graph patterns for the q_vanilla_accelerator accelerator"""

from tvm.relay.dataflow_pattern import is_op, wildcard, is_constant
import tvm

def conv2d_pattern():
    pattern = is_op("nn.conv2d")(wildcard(), wildcard())
    pattern = pattern.has_attr({"strides": [1, 1], "groups": 1})
    print("nn pattern")
    return pattern



def qnn_conv2d_pattern():
    
    pattern = is_op("qnn.conv2d")(wildcard(), wildcard(), is_constant(), is_constant(), is_constant(), is_constant(),)
    
    pattern = pattern.has_attr({"strides": [1, 1], "groups": 1})
    # qnn_conv2d = qnn_conv2d.has_attr({"strides": [1, 1], "groups": 1})

    # pattern = is_op("add")(qnn_conv2d, wildcard())

    print("qnn_conv2d_pattern_after")

    # req = is_op("qnn.requantize")(
    #     add, is_constant(), is_constant(), is_constant(), is_constant()
    # )
    # pattern = is_op("clip")(add)

    return pattern


def qnn_conv2d_add_pattern():
    
    qnn_conv2d = is_op("qnn.conv2d")(wildcard(), wildcard(), is_constant(),
                         is_constant(), is_constant(), is_constant(),)
    

    qnn_conv2d = qnn_conv2d.has_attr({"strides": [1, 1], "groups": 1})

    pattern = is_op("add")(qnn_conv2d, wildcard())


    return pattern   



def dense_pattern():
    pattern = is_op("nn.dense")(wildcard(), wildcard())
    return pattern

#padding = is_op("nn.pad")(wildcard(), is_constant())


# def qnn_conv2d_pattern() -> tvm.relay.dataflow_pattern.DFPattern:
#     """
#     This function creates the pattern for qnn.conv2D with optional fused RELU activation.
#     """
#     optional_pad = is_op("nn.pad")(wildcard(), is_constant())
#     qnn_conv2d = is_op("qnn.conv2d")(
#         optional_pad | wildcard(),
#         is_constant(),
#         is_constant(),
#         is_constant(),
#         is_constant(),
#         is_constant(),
#     ).has_attr({"kernel_layout": "HWIO"})
#     bias_add = is_op("nn.bias_add")(qnn_conv2d, is_constant())
#     req = is_op("qnn.requantize")(
#         bias_add, is_constant(), is_constant(), is_constant(), is_constant()
#     )
#     clip_or_req = req.optional(is_op("clip"))
#     return clip_or_req

