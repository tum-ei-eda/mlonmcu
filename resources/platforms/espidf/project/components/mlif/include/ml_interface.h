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
#ifndef ML_INTERFACE_H
#define ML_INTERFACE_H

#include <stddef.h>
#include <stdbool.h>

#ifndef NUM_RUNS
#define NUM_RUNS 1
#endif  /* NUM_RUNS */

#ifdef __cplusplus
extern "C" {
#endif

void mlif_run(void);

// These can be overridden by use code.

// Provides input data for the model. The default implementation retrieves input from
// the global variables below and fills the model input with mlif_process_input.
int mlif_request_input(void *model_input_ptr, size_t model_input_sz, bool *new_);
// Is called when the output data is available. The default implementation
int mlif_handle_result(void *model_output_ptr, size_t model_output_sz);

// Callback for any preprocessing on the input data. Responsible for copying the data.
int mlif_process_input(const void *in_data, size_t in_size, void *model_input_ptr, size_t model_input_sz);
// Callback for any postprocessing on the output data. The default implementation prints
// the output and verifies consistency with the expected output.
int mlif_process_output(void *model_output_ptr, size_t model_output_sz, const void *expected_out_data,
                         size_t expected_out_size);

extern const int num_data_buffers_in;
extern const int num_data_buffers_out;
extern const unsigned char * const data_buffers_in[];
extern const unsigned char * const data_buffers_out[];
extern const size_t data_size_in[];
extern const size_t data_size_out[];

extern const int num;

#ifdef __cplusplus
}
#endif

#endif
