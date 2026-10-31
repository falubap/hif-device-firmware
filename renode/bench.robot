*** Comments ***
Step 6 benchmark for a MODE=bench image. A CPU hook on bench_mark() appends "<id>,<executed
instructions>;" to ${MARKS} each time the firmware calls it. The hook runs exactly when the CPU
reaches that address, so the counts are exact to the instruction; pausing on UART lines is not
(the emulator stops at its next synchronisation point, up to millions of instructions later).
Driven by python/measure.py:
    renode-test renode/bench.robot --variable ELF:<abs> --variable UART_LOG:<abs> --variable MARKS:<abs>

*** Settings ***
Library    String

*** Variables ***
${ELF}         ${CURDIR}/../firmware/build/hif.elf
${UART_LOG}    ${CURDIR}/../data/renode/bench.log
${MARKS}       ${CURDIR}/../data/renode/bench_marks.txt
${TIMEOUT}     900

*** Test Cases ***
Count Instructions
    Execute Command          mach create
    Execute Command          machine LoadPlatformDescription @platforms/boards/stm32f4_discovery-kit.repl
    Execute Command          sysbus LoadELF @${ELF}
    Execute Command          sysbus.usart2 CreateFileBackend @${UART_LOG} true
    ${addr}=                 Execute Command    sysbus GetSymbolAddress "bench_mark"
    ${addr}=                 Strip String    ${addr}
    Execute Command          sysbus.cpu AddHook ${addr} "f = open(r'${MARKS}', 'a'); f.write('%d,%d;' % (self.GetRegister(0).RawValue, self.ExecutedInstructions)); f.close()"
    Create Terminal Tester   sysbus.usart2    timeout=${TIMEOUT}
    Start Emulation
    Wait For Line On Uart    DONE
