#include <stdlib.h>
#include <memory.h>

#include "printing.h"
#include "exit.h"

void mlif_process_output(void *model_output_ptr, size_t model_output_sz, const void *expected_out_data,
                         size_t expected_out_size) {
#ifdef _DEBUG
  if (model_output_sz >= 4) {
    DBGPRINTF("MLIF: First float of output: %f\n", *(float *)model_output_ptr);
  }
  DBGPRINTF("MLIF: Model output data: ");
  for (size_t i = 0; i < model_output_sz; i++) {
    DBGPRINTF("\\x%02X", ((unsigned char *)model_output_ptr)[i]);
    fflush(0);
  }
  DBGPRINTF("\n");
#endif

  if (model_output_sz == expected_out_size && memcmp(model_output_ptr, expected_out_data, model_output_sz) == 0) {
    DBGPRINTF("MLIF: Output data matches expected data\n");
  } else {
    DBGPRINTF("MLIF: Wrong output data!\n");
    exit(EXIT_MLIF_MISSMATCH);
  }
}
