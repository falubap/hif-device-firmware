# Build the HIF firmware for the STM32F407 (Cortex-M4F, hard float).
#   make            build build/hif.elf
#   make OPT=-O2    same, optimised for speed (the sweep varies this)
#   make size       flash/RAM usage
#   make run        open it in Renode (GUI, UART analyzer window)

CROSS   ?= arm-none-eabi-
CC      := $(CROSS)gcc
SIZE    := $(CROSS)size

BUILD   := build
TARGET  := $(BUILD)/hif.elf
OPT     ?= -Os

CMSIS_DSP  := third_party/CMSIS-DSP
CMSIS_CORE := third_party/CMSIS_6/CMSIS/Core

SRCS := src/startup.c src/uart.c src/main.c
OBJS := $(SRCS:src/%.c=$(BUILD)/%.o)

ARCH    := -mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16
CFLAGS  := $(ARCH) $(OPT) -std=c11 -Wall -Wextra -g \
           -ffunction-sections -fdata-sections -fstack-usage \
           -Isrc -Iinclude -I$(CMSIS_DSP)/Include -I$(CMSIS_CORE)/Include
LDFLAGS := $(ARCH) -T linker.ld -nostartfiles --specs=nano.specs --specs=nosys.specs \
           -Wl,--gc-sections -Wl,-Map=$(BUILD)/hif.map -Wl,--print-memory-usage

.PHONY: all size run clean

all: $(TARGET)

$(TARGET): $(OBJS) linker.ld
	$(CC) $(LDFLAGS) $(OBJS) -o $@

$(BUILD)/%.o: src/%.c | $(BUILD)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

$(BUILD):
	mkdir -p $@

size: $(TARGET)
	$(SIZE) $<

run: $(TARGET)
	renode ../renode/hif.resc

clean:
	rm -rf $(BUILD)

-include $(OBJS:.o=.d)
