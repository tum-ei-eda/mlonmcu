/*
 * Copyright (c) 2022 TUM Department of Electrical and Computer Engineering.
 *
 * This file is part of MLonMCU.
 * See https://github.com/tum-ei-eda/mlonmcu.git for further info.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "ml_interface.h"
#include "tvm_wrapper.h"

void mlif_init() {
  TVMWrap_Init();
}

void *mlif_input_ptr(int index) {
  return TVMWrap_GetInputPtr(index);

}

size_t mlif_input_size(int index) {
  return TVMWrap_GetInputSize(index);
}

size_t mlif_inputs() {
  return TVMWrap_GetNumInputs();
}

void mlif_invoke() {
  TVMWrap_Run();
}

void *mlif_output_ptr(int index) {
  return TVMWrap_GetOutputPtr(index);
}

size_t mlif_output_size(int index) {
  return TVMWrap_GetOutputSize(index);
}

size_t mlif_outputs() {
  return TVMWrap_GetNumOutputs();
}
