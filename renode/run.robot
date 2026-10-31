*** Comments ***
Run a firmware image headless until it prints DONE, saving everything USART2 prints.
Driven by python/run_device.py:
    renode-test renode/run.robot --variable ELF:<abs path> --variable UART_LOG:<abs path>

*** Variables ***
${ELF}         ${CURDIR}/../firmware/build/hif.elf
${UART_LOG}    ${CURDIR}/../data/uart.log
${TIMEOUT}     600

*** Test Cases ***
Run Firmware
    Execute Command          mach create
    Execute Command          machine LoadPlatformDescription @platforms/boards/stm32f4_discovery-kit.repl
    Execute Command          sysbus LoadELF @${ELF}
    Execute Command          sysbus.usart2 CreateFileBackend @${UART_LOG} true
    Create Terminal Tester   sysbus.usart2    timeout=${TIMEOUT}
    Start Emulation
    Wait For Line On Uart    DONE
