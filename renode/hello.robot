*** Comments ***
Step 1 check, headless: the firmware boots, prints HELLO and survives a hardware float op.
Run: renode-test renode/hello.robot

*** Variables ***
${ELF}    ${CURDIR}/../firmware/build/hif.elf

*** Test Cases ***
Boots And Prints Hello
    Execute Command          mach create
    Execute Command          machine LoadPlatformDescription @platforms/boards/stm32f4_discovery-kit.repl
    Execute Command          sysbus LoadELF @${ELF}
    Create Terminal Tester   sysbus.usart2    timeout=5
    Start Emulation
    Wait For Line On Uart    HELLO
    Wait For Line On Uart    FPU OK
