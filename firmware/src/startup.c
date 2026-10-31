/* Reset handler and vector table for the STM32F407 (Cortex-M4F).
 *
 * Only the 16 core exceptions are listed: the firmware enables no peripheral
 * interrupts, so the device IRQ entries would never be fetched. */

#include <stdint.h>

#include "startup.h"
#include "uart.h"

extern uint32_t _estack, _sstack;
extern uint32_t _sidata, _sdata, _edata, _sbss, _ebss;

/* Stack painting: the free stack is filled with this word at boot; the deepest word that no
 * longer holds it marks the stack's high-water mark. */
#define STACK_PAINT 0xDEADBEEFu

int main(void);

/* Coprocessor Access Control Register (System Control Block). */
#define SCB_CPACR (*(volatile uint32_t *)0xE000ED88u)

void Reset_Handler(void)
{
    /* Full access to CP10/CP11 (the FPU). This must happen before any float
     * instruction runs, or the first one raises a UsageFault. */
    SCB_CPACR |= (0xFu << 20);
    __asm volatile("dsb\n\tisb" ::: "memory");

    /* Paint [_sstack, sp - 64): everything below this function's own frame. */
    uint32_t sp;
    __asm volatile("mov %0, sp" : "=r"(sp));
    for (uint32_t *p = &_sstack; (uint32_t)p < sp - 64; p++)
        *p = STACK_PAINT;

    uint32_t *src = &_sidata;
    for (uint32_t *dst = &_sdata; dst < &_edata;)
        *dst++ = *src++;
    for (uint32_t *dst = &_sbss; dst < &_ebss;)
        *dst++ = 0;

    main();
    for (;;)
        ;
}

uint32_t stack_peak_bytes(void)
{
    const uint32_t *p = &_sstack;
    while (p < &_estack && *p == STACK_PAINT)
        p++;
    return (uint32_t)((const uint8_t *)&_estack - (const uint8_t *)p);
}

void HardFault_Handler(void)
{
    uart_puts("HARDFAULT\r\n");
    for (;;)
        ;
}

void Default_Handler(void)
{
    uart_puts("UNHANDLED EXCEPTION\r\n");
    for (;;)
        ;
}

void NMI_Handler(void) __attribute__((weak, alias("Default_Handler")));
void MemManage_Handler(void) __attribute__((weak, alias("Default_Handler")));
void BusFault_Handler(void) __attribute__((weak, alias("Default_Handler")));
void UsageFault_Handler(void) __attribute__((weak, alias("Default_Handler")));
void SVC_Handler(void) __attribute__((weak, alias("Default_Handler")));
void DebugMon_Handler(void) __attribute__((weak, alias("Default_Handler")));
void PendSV_Handler(void) __attribute__((weak, alias("Default_Handler")));
void SysTick_Handler(void) __attribute__((weak, alias("Default_Handler")));

__attribute__((section(".isr_vector"), used))
void (*const vector_table[16])(void) = {
    (void (*)(void))&_estack,
    Reset_Handler,
    NMI_Handler,
    HardFault_Handler,
    MemManage_Handler,
    BusFault_Handler,
    UsageFault_Handler,
    0, 0, 0, 0,
    SVC_Handler,
    DebugMon_Handler,
    0,
    PendSV_Handler,
    SysTick_Handler,
};
