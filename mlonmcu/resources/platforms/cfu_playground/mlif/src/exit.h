#ifndef SUPPORT_EXIT_H
#define SUPPORT_EXIT_H

#include <stdlib.h>

#define EXIT_MLIF_BASE (0x10)
#define EXIT_MLIF_INVALID_SIZE (EXIT_MLIF_BASE + 1)
#define EXIT_MLIF_MISSMATCH (EXIT_MLIF_BASE + 2)

void mlonmcu_exit(int status);

#endif  // SUPPORT_EXIT_H
