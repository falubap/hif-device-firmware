# Build the HIF firmware for the STM32F407 (Cortex-M4F, hard float).
#   make                 build build/hif.elf (MODE=hello: boot check)
#   make MODE=model      Step 3: model on the reference features in include/test_vectors.h
#   make OPT=-O2         same, optimised for speed (the sweep varies this)
#   make size            flash/RAM usage
#   make run             open it in Renode (GUI, UART analyzer window)
# Changing MODE or OPT rebuilds everything (see $(BUILD)/flags).

CROSS   ?= arm-none-eabi-
CC      := $(CROSS)gcc
SIZE    := $(CROSS)size

BUILD   := build
TARGET  := $(BUILD)/hif.elf
OPT     ?= -Os
MODE    ?= hello

CMSIS_DSP   := third_party/CMSIS-DSP
CMSIS_CORE  := third_party/CMSIS_6/CMSIS/Core
# emlearn's runtime headers (eml_net.h etc.), from the Python package that generated model.h.
EMLEARN_INC ?= $(shell ../.venv/bin/python -c 'import emlearn; print(emlearn.includedir)' 2>/dev/null)

SRCS := src/startup.c src/uart.c src/main.c
OBJS := $(SRCS:src/%.c=$(BUILD)/%.o)

MODE_DEFINE := -DHIF_MODE_$(shell echo $(MODE) | tr a-z A-Z)

ARCH    := -mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16
# Generated headers (include/) and third-party code are -isystem: their warnings are not ours.
CFLAGS  := $(ARCH) $(OPT) -std=c11 -Wall -Wextra -g $(MODE_DEFINE) \
           -ffunction-sections -fdata-sections -fstack-usage \
           -Isrc -isystem include -isystem $(EMLEARN_INC) \
           -isystem $(CMSIS_DSP)/Include -isystem $(CMSIS_CORE)/Include
LDFLAGS := $(ARCH) -T linker.ld -nostartfiles --specs=nano.specs --specs=nosys.specs \
           -Wl,--gc-sections -Wl,-Map=$(BUILD)/hif.map -Wl,--print-memory-usage

.PHONY: all size run clean FORCE

all: $(TARGET)

$(TARGET): $(OBJS) linker.ld
	$(CC) $(LDFLAGS) $(OBJS) -o $@ -lm

# -MD, not -MMD: the generated headers sit on an -isystem path, and -MMD would drop them from
# the dependencies, so a new model.h or test_vectors.h would not trigger a rebuild.
$(BUILD)/%.o: src/%.c $(BUILD)/flags | $(BUILD)
	$(CC) $(CFLAGS) -MD -MP -c $< -o $@

# Rewritten only when the flags change, so objects rebuild on a MODE or OPT switch.
$(BUILD)/flags: FORCE | $(BUILD)
	@echo '$(CFLAGS)' | cmp -s - $@ || echo '$(CFLAGS)' > $@

$(BUILD):
	mkdir -p $@

size: $(TARGET)
	$(SIZE) $<

run: $(TARGET)
	renode ../renode/hif.resc

clean:
	rm -rf $(BUILD)

-include $(OBJS:.o=.d)
