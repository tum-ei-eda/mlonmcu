#ifndef SUPPORT_PRINTING_H
#define SUPPORT_PRINTING_H

#include "target.h"
// void mlonmcu_printf(const char* format, ...);

#ifdef STRIP_STRINGS
#define mlonmcu_printf(format, ...)
#else
#define mlonmcu_printf target_printf
#endif  // STRIP_STRINGS

#ifdef _DEBUG
#define DBGPRINTF(format, ...) mlonmcu_printf(format, ##__VA_ARGS__)
#else
#define DBGPRINTF(format, ...)
#endif

#endif
