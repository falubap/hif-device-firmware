# Measurement method

How each number in `results/measurements.csv` is obtained (`python/measure.py`), and what it
does and does not mean.

## 1. What is measured

| Metric | How | Notes |
| --- | --- | --- |
| Flash | `text` + `data` from the linker map, per module | Test data (`hif_tv_*`, `hif_bench_*`) excluded |
| Static RAM | `data` + `bss` from the linker map, per module | Buffers are static and sized by the configuration |
| Stack | Peak by stack painting; per-function frames from `-fstack-usage` | The painted peak covers the whole benchmark run |
| Input buffer | Computed: raw signals the features read × 2 windows × 2560 samples × 4 B | Not in the simulated RAM (the windows sit in flash as test data); a real device must hold it |
| Executed instructions | Renode's counter, read by a CPU hook at marker calls | Per window (extraction) and per feature vector (inference), plus extraction split by stage |
| Correctness | Every benchmark build re-checks its predictions against scikit-learn | A measurement is recorded only if all agree |

Modules: **extraction** (`features.c`, `dsp2560.c`), **CMSIS-DSP** (FFT code and tables),
**model** (`model.c`, the generated model), **libc** (newlib libc/libm, libgcc), **app**
(startup, UART, main).

### Instruction counts

The benchmark firmware (`make MODE=bench`) calls `bench_mark(id)`:

1. before and after 2 passes of feature extraction over the raw windows in the build (5–7);
2. around inference on the reference features of all 1645 test rows;
3. after each stage of one extraction: DFT, Stockwell, wavelet, statistics.

`renode/bench.robot` sets a CPU hook on `bench_mark`. The hook runs when the CPU reaches that
address and records `ExecutedInstructions`, so counts are exact to the instruction. That was
checked on a loop of known length: 1000 iterations × 6 instructions came out as 6,005
instructions between marks, the loop plus the call.

The obvious alternative, pausing the emulation when a marker line appears on the UART, is not
precise. Renode stops at its next synchronisation point, which can be millions of instructions
past the marker. The first version of the benchmark used it, and it reported 0 instructions for
whole extraction stages.

Per window = (instructions between marks) ÷ (windows processed). Extraction is nearly
data-independent: the profiled single window and the mean over 10 differ by less than 0.001%.
Inference is not: a tree's cost depends on the path taken, so it is averaged over all 1645 test
rows.

## 2. Why an instruction count is not an execution time

Renode is a **functional** simulator. It executes the instructions correctly and counts them, but
it does not model how long each one takes. On a real STM32F407 the time for the same instruction
stream depends on:

- **Pipeline effects.** The Cortex-M4 has a 3-stage pipeline. A taken branch flushes it and
  costs 1–3 extra cycles. Branch-heavy code, such as the inlined decision trees (a chain of
  `if`s), loses more to this than straight-line loops do.
- **Flash wait states.** At 168 MHz the STM32F4 flash needs 5 wait states. The ART accelerator
  (prefetch plus a small instruction cache) hides them for sequential code. It cannot hide them
  for code too large for the cache that jumps around: the 113 KB forest is exactly that. Constant
  tables read from flash (CMSIS twiddles, filter taps) have the same problem.
- **Instruction latency.** Instructions do not take one cycle each. A load takes 2 cycles unless
  pipelined. A float multiply-accumulate (`VMLA`) takes 3. A float divide or square root
  (`VDIV`, `VSQRT`) takes 14. Calls to `sinf`, `logf` or `atan2f` run tens of instructions with
  their own mix of costs.
- **Memory system.** SRAM runs without wait states, but the bus matrix is shared with DMA. A
  real device would have ADC DMA writing samples while the CPU computes.
- **What the simulation leaves out.** On real hardware, UART output at 115200 baud takes about
  87 µs per character, and the CPU busy-waits for it. Interrupts would also take time. The
  benchmark brackets contain no I/O, but anything that does printing would be far slower on a
  real board than its instruction count suggests.

So instruction counts are a sound basis for **comparing configurations** on the same code base:
more features, a bigger forest, `-O2` vs `-Os`. They are **not** milliseconds.
`est_ms_168mhz` divides instructions by 168 MHz, which assumes 1 instruction per cycle; it is
labelled as a rough estimate everywhere it appears. Real DSP-heavy code on this part typically
needs more than 1 cycle per instruction once wait states and FPU latencies count, so the
estimate is optimistic. Measuring on a real board with the DWT cycle counter would settle it.
