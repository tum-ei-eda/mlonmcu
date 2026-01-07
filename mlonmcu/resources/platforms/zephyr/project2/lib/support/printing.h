#ifndef SUPPORT_PRINTING_H
#define SUPPORT_PRINTING_H

#ifdef _DEBUG
#include <stdio.h>
#define DBGPRINTF(format, ...) printf(format, ##__VA_ARGS__)
#else
#define DBGPRINTF(format, ...)
#endif

#endif
