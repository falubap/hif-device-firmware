#include "uart.h"

int main(void)
{
    uart_init();
    uart_puts("HELLO\r\n");

    /* One hardware float operation, so a missing or disabled FPU shows up
     * here as HARDFAULT instead of later inside the feature code. volatile
     * stops the compiler from folding it at build time. */
    volatile float a = 1.5f, b = 2.25f;
    uart_puts(a * b == 3.375f ? "FPU OK\r\n" : "FPU WRONG\r\n");

    for (;;)
        ;
}
