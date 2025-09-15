#include "ml_interface.h"

#include "model.cc.h"

void mlif_run() {
  model_init();

  size_t input_num = 0;
  size_t remaining = NUM_RUNS;
  while (mlif_request_input(model_input_ptr(input_num), model_input_size(input_num)) || remaining) {
    if (input_num == model_inputs() - 1) {
      model_invoke();
      for (size_t i = 0; i < model_outputs(); i++) {
        mlif_handle_result(model_output_ptr(i), model_output_size(i));
      }
      input_num = 0;
      remaining = remaining > 0 ? remaining - 1 : 0;
    } else {
      input_num++;
    }
  }
}
