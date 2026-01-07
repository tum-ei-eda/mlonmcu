#include "fake_input.h"
#include <stdint.h>

void FakeInput_GetData(void *out, size_t len) {
  volatile uint8_t *p = 0xc0000000;

  for (size_t i = 0; i < len; i++) {
    *(uint8_t *)out = *(p + i);
  }
}
