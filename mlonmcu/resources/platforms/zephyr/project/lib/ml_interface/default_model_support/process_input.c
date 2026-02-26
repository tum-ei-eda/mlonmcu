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
