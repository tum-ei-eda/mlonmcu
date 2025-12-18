#include "mlonmcu.h"
#include "target.h"
#include "exit.h"
#include "bench.h"
#include "printing.h"
#include <stdio.h>

int main() {
  int ret = 0;
  // pre
  target_init();
  mlonmcu_printf("Program start.\n");

  // main
  start_bench(TOTAL);
  mlonmcu_printf("Hello World.\n");
  stop_bench(TOTAL);

  // post
  print_bench(TOTAL);
  mlonmcu_printf("Program finish.\n");
  target_deinit();

  mlonmcu_exit(ret);
  return ret;
}
