#ifndef STARTUP_H
#define STARTUP_H

#include <stdint.h>

/* Deepest stack use since reset, in bytes (stack painting; see startup.c). */
uint32_t stack_peak_bytes(void);

#endif
