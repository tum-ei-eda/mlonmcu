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
#include <stdlib.h>
#include <memory.h>

#include "printing.h"
#include "exit.h"

void mlif_process_input(const void *in_data, size_t in_size, void *model_input_ptr, size_t model_input_sz) {
  if (in_size != 0) {
    if (in_size != model_input_sz) {
      DBGPRINTF("MLIF: Given input size (%lu) does not match model input buffer size (%lu)!\n", in_size,
                model_input_sz);
      exit(EXIT_MLIF_INVALID_SIZE);
    }
  }

  memcpy(model_input_ptr, in_data, in_size);
}
