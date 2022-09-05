#include "ml_interface.h"
#include "printing.h"

#include <string.h>
#include <stdlib.h>

__attribute__((weak)) bool mlif_request_input(void *model_input_ptr, size_t model_input_sz) {
  static int num_done = 0;
  if (num_done == num_data_buffers_in) {
    static bool run_once = true;
    if (num_data_buffers_in == 0 && run_once) {
      // Minimal run. Just run the model without data.
      run_once = false;
      return true;
    }
    return false;
  }

  mlif_process_input(data_buffers_in[num_done], data_size_in[num_done], model_input_ptr, model_input_sz);
  num_done++;
  return true;
}

__attribute__((weak)) void mlif_handle_result(void *model_output_ptr, size_t model_output_sz) {
  static int num_done = 0;

  if (num_data_buffers_out == 0) {
    return;
  }

  if (num_done < num_data_buffers_out) {
    mlif_process_output(model_output_ptr, model_output_sz, data_buffers_out[num_done], data_size_out[num_done]);
  }

  num_done++;
}
