#ifndef UART_H
#define UART_H

/* Blocking output on USART2 (PA2 TX), 115200 8N1 from the 16 MHz HSI clock. */

void uart_init(void);
void uart_putc(char c);
void uart_puts(const char *s);

#endif
