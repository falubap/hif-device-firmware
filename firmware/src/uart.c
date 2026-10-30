/* USART2 driver written against the registers (RM0090), no vendor HAL. */

#include <stdint.h>

#include "uart.h"

#define REG(addr) (*(volatile uint32_t *)(addr))

#define RCC_BASE        0x40023800u
#define RCC_AHB1ENR     REG(RCC_BASE + 0x30)
#define RCC_APB1ENR     REG(RCC_BASE + 0x40)
#define RCC_GPIOAEN     (1u << 0)
#define RCC_USART2EN    (1u << 17)

#define GPIOA_BASE      0x40020000u
#define GPIOA_MODER     REG(GPIOA_BASE + 0x00)
#define GPIOA_AFRL      REG(GPIOA_BASE + 0x20)

#define USART2_BASE     0x40004400u
#define USART2_SR       REG(USART2_BASE + 0x00)
#define USART2_DR       REG(USART2_BASE + 0x04)
#define USART2_BRR      REG(USART2_BASE + 0x08)
#define USART2_CR1      REG(USART2_BASE + 0x0C)
#define USART_SR_TXE    (1u << 7)
#define USART_CR1_UE    (1u << 13)
#define USART_CR1_TE    (1u << 3)

#define HSI_HZ          16000000u
#define BAUD            115200u

void uart_init(void)
{
    RCC_AHB1ENR |= RCC_GPIOAEN;
    RCC_APB1ENR |= RCC_USART2EN;

    /* PA2 -> alternate function 7 (USART2_TX). */
    GPIOA_MODER = (GPIOA_MODER & ~(3u << 4)) | (2u << 4);
    GPIOA_AFRL = (GPIOA_AFRL & ~(0xFu << 8)) | (7u << 8);

    USART2_BRR = (HSI_HZ + BAUD / 2) / BAUD;
    USART2_CR1 = USART_CR1_UE | USART_CR1_TE;
}

void uart_putc(char c)
{
    while (!(USART2_SR & USART_SR_TXE))
        ;
    USART2_DR = (uint8_t)c;
}

void uart_puts(const char *s)
{
    while (*s)
        uart_putc(*s++);
}
