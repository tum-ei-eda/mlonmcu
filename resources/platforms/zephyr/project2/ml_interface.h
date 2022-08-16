#ifndef TARGETLIB_ML_INTERFACE_H
#define TARGETLIB_ML_INTERFACE_H

#include <stddef.h>
#include <stdbool.h>

#ifndef NUM_RUNS
#define NUM_RUNS 1
#endif /* NUM_RUNS */

#ifdef __cplusplus
extern "C" {
#endif

// This runs the ML model using the callbacks above.
// The default implementation will run with garbage data.
void mlif_run();

// These can be overridden by use code.

// Provides input data for the model. The default implementation retrieves input from
// the global variables below and fills the model input with mlif_process_input.
bool mlif_request_input(void *model_input_ptr, size_t model_input_sz);
// Is called when the output data is available. The default implementation
void mlif_handle_result(void *model_output_ptr, size_t model_output_sz);

// Callback for any preprocessing on the input data. Responsible for copying the data.
void mlif_process_input(const void *in_data, size_t in_size, void *model_input_ptr, size_t model_input_sz);
// Callback for any postprocessing on the output data. The default implementation prints
// the output and verifies consistency with the expected output.
void mlif_process_output(void *model_output_ptr, size_t model_output_sz, const void *expected_out_data,
                         size_t expected_out_size);

extern const int num_data_buffers_in;
extern const int num_data_buffers_out;
extern const unsigned char *const data_buffers_in[];
extern const unsigned char *const data_buffers_out[];
extern const size_t data_size_in[];
extern const size_t data_size_out[];

extern const int num;

#ifdef __cplusplus
}
#endif

#endif
