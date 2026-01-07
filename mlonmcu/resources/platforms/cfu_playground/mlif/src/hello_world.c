#include <stdio.h>
#include "printing.h"

int mlonmcu_init() {
  return 0;
}
int mlonmcu_deinit() {
  return 0;
}
int mlonmcu_run() {
  mlonmcu_printf("Hello World!\n");
  return 0;
}
int mlonmcu_check() {
  return 0;
}
