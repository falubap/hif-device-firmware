/* STM32F407VG: 1 MB flash, 128 KB SRAM (+64 KB CCM at 0x10000000, unused). */

ENTRY(Reset_Handler)

MEMORY
{
    FLASH (rx)  : ORIGIN = 0x08000000, LENGTH = 1024K
    RAM   (rwx) : ORIGIN = 0x20000000, LENGTH = 128K
}

/* The stack grows down from the top of RAM. The link fails if static data
 * leaves less than this much room for it. */
_estack = ORIGIN(RAM) + LENGTH(RAM);
_min_stack_size = 0x2000;

SECTIONS
{
    .isr_vector :
    {
        . = ALIGN(4);
        KEEP(*(.isr_vector))
        . = ALIGN(4);
    } > FLASH

    .text :
    {
        . = ALIGN(4);
        *(.text*)
        *(.rodata*)
        . = ALIGN(4);
    } > FLASH

    .ARM.exidx :
    {
        *(.ARM.exidx* .gnu.linkonce.armexidx.*)
    } > FLASH

    /* Initial values of .data, copied to RAM by Reset_Handler. */
    _sidata = LOADADDR(.data);

    .data :
    {
        . = ALIGN(4);
        _sdata = .;
        *(.data*)
        . = ALIGN(4);
        _edata = .;
    } > RAM AT > FLASH

    .bss (NOLOAD) :
    {
        . = ALIGN(4);
        _sbss = .;
        *(.bss*)
        *(COMMON)
        . = ALIGN(4);
        _ebss = .;
    } > RAM

    /* The stack owns everything from here to _estack. It is not a section, so
     * `size` reports bss as static data only; stack painting (Step 6) fills
     * [_sstack, _estack). */
    _sstack = ALIGN(_ebss, 8);
    ASSERT(_sstack + _min_stack_size <= _estack, "not enough RAM left for the stack")

    /DISCARD/ :
    {
        *(.ARM.attributes)
    }
}
