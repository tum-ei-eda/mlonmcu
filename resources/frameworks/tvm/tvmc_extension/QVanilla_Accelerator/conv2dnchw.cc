/*
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
*/

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

// TODO(mjklaiber): leverage pragma import_c in the future
#ifdef __cplusplus
extern "C"
#endif

    /*!
     * \brief Conv2D function for mock-accelerator examples. Limited to same-padded Conv2D with
     * stride (1,1) and datatype float. \param ifmap Pointer to input feature map data of size
     * iw*ih*ic*sizeof(float). \param weights Pointer to weight data of size
     * kh*kw*ic**oc*sizeof(float). \param result Pointer to output feature map data of size
     * iw*ih*oc*sizeof(float). \param oc Number of channels of output feature map. \param iw Width
     * of input feature map, ifmap. \param ih Height of input feature map, ifmap. \param ic Number
     * of channels of input feature map. \param kh Height of convolution kernels. \param kw Width of
     * convolution kernels.
     *
     * \return error code
     *
     */

// uint32_t swap_endian(int a) 
// {
//   uint32_t b = (uint32_t) a;
//   uint32_t b0 = b & 0x000000ff;
//   uint32_t b1 = (b & 0x0000ff00) >> 8;
//   uint32_t b2 = (b & 0x00ff0000) >> 16;
//   uint32_t b3 = (b & 0xff000000) >> 24;
//   uint32_t o;
//   o = (b0 << 24) | (b1 << 16) | (b2 << 8) | b3;
//   return o;
// }

int q_vanilla_accelerator_conv2dnchw(int8_t* q_vanilla_accelerator_0_i0, int8_t* q_vanilla_accelerator_0_i1, int32_t* bias_data, int32_t* compute,
                                      int32_t oc, int32_t iw, int32_t ih, int32_t ic, int32_t kh, int32_t kw, int32_t i_zp, int32_t k_zp) {


  
  //printf("start writing...\n");
  *(int32_t**)0x70000000 = (int32_t*)q_vanilla_accelerator_0_i0;
  *(int32_t**)0x70000004 = (int32_t*)q_vanilla_accelerator_0_i1;
  *(int32_t**)0x70000008 = (int32_t*)bias_data;
  *(int32_t**)0x7000000c = (int32_t*)compute;

  *(int32_t*)0x70000010 = oc;
  *(int32_t*)0x70000014 = iw;
  *(int32_t*)0x70000018 = ih;
  *(int32_t*)0x7000001c = ic;
  *(int32_t*)0x70000020 = kh;
  *(int32_t*)0x70000024 = kw;
  *(int32_t*)0x70000028 = i_zp;
  *(int32_t*)0x7000002c = k_zp;

  //issue start signal
  //printf("issue start ...\n");
  *(int32_t*)0x70000030 = 0x00000001;
  
  

  return 0;
}
