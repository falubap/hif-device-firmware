/* HIF region locator firmware.
 *
 * Build modes (make MODE=...):
 *   hello     boot check: HELLO, one hardware float op, FPU OK
 *   model     Step 3: predict from the Python reference features in test_vectors.h, so the
 *             model is checked on the target before feature extraction exists
 *   features  Step 4: extract the features of each raw window in test_vectors.h
 *   full      Step 5: raw window -> features -> model, for each window in test_vectors.h
 *   bench     Step 6: instruction counts. Marks (bench_mark) around BENCH_EXTRACT_REPEAT passes
 *             of feature extraction over the raw windows, then inference over every test row's
 *             reference features, then one window again with a mark after each extraction
 *             stage. renode/bench.robot hooks bench_mark and logs Renode's instruction counter
 *             at each call, exact to the instruction.
 *
 * Output lines are parsed by python/run_device.py and python/measure.py:
 *   P,<test row>,<predicted region>
 *   F,<test row>,<feature 0 bits>,...   (float32 bit patterns, hex)
 *   BENCH,<extractions>,<inferences>,<inferences that disagree with Python>
 *   STACK,<peak stack bytes>
 *   DONE
 */

#include <string.h>

#include "startup.h"
#include "uart.h"

#if defined(HIF_MODE_FEATURES) || defined(HIF_MODE_FULL) || defined(HIF_MODE_BENCH)
#define HIF_EXTRACT
#endif

#if defined(HIF_MODE_MODEL) || defined(HIF_EXTRACT)
/* model.h is included by model.c only: emlearn's code defines external functions. */
#include "model_api.h"
#include "test_vectors.h"

#define HIF_N_FEATURES HIF_TV_N_FEATURES

static void print_prediction(int k, int pred)
{
    uart_puts("P,");
    uart_putu(hif_tv_index[k]);
    uart_putc(',');
    uart_putu((uint32_t)pred);
    uart_puts("\r\n");
}
#endif

#if defined(HIF_EXTRACT)
#include "feature_config.h"

_Static_assert(HIF_CFG_N_FEATURES == HIF_N_FEATURES, "feature_config.h and test_vectors.h disagree");
_Static_assert(HIF_TV_HAS_RAW, "features mode needs raw windows (export_vectors.py --batch)");
#endif

#if defined(HIF_MODE_MODEL)
static void run_model(void)
{
    for (int k = 0; k < HIF_TV_COUNT; k++)
        print_prediction(k, hif_predict(hif_tv_features[k]));
}
#endif

#if defined(HIF_EXTRACT)
static void print_features(int k, const float *feats)
{
    uart_puts("F,");
    uart_putu(hif_tv_index[k]);
    for (int i = 0; i < HIF_N_FEATURES; i++) {
        uint32_t bits;
        memcpy(&bits, &feats[i], sizeof bits);
        uart_putc(',');
        uart_puthex(bits);
    }
    uart_puts("\r\n");
}

#if defined(HIF_MODE_FEATURES) || defined(HIF_MODE_FULL)
/* Features of every window; with predict, also the model's region (full pipeline). */
static void run_extract(int predict)
{
    static float feats[HIF_N_FEATURES];
    hif_features_init();
    for (int k = 0; k < HIF_TV_COUNT; k++) {
        hif_extract_features(hif_tv_x[k], hif_tv_phase[k], feats);
        print_features(k, feats);
        if (predict)
            print_prediction(k, hif_predict(feats));
    }
}
#endif
#endif

#if defined(HIF_MODE_BENCH)
#define BENCH_EXTRACT_REPEAT 2
_Static_assert(HIF_TV_HAS_BENCH, "bench mode needs export_vectors.py --bench");

/* Mark ids, in order (python/measure.py reads them by number). */
enum {
    MARK_EXTRACT, MARK_INFER, MARK_END,
    MARK_PROFILE, MARK_DFT, MARK_STOCKWELL, MARK_WAVELET, MARK_STATS,
};

/* Renode runs a hook when the CPU reaches this function and records (id, executed
 * instructions). It must stay a real call: noinline, and id kept live in r0. */
__attribute__((noinline)) void bench_mark(int id)
{
    __asm volatile("" : : "r"(id));
}

static void profile_mark(void)
{
    static int next = MARK_DFT;
    bench_mark(next++);
}

static void run_bench(void)
{
    static float feats[HIF_TV_COUNT][HIF_N_FEATURES];
    hif_features_init();

    bench_mark(MARK_EXTRACT);
    for (int r = 0; r < BENCH_EXTRACT_REPEAT; r++)
        for (int k = 0; k < HIF_TV_COUNT; k++)
            hif_extract_features(hif_tv_x[k], hif_tv_phase[k], feats[k]);
    bench_mark(MARK_INFER);
    uint32_t wrong = 0;
    for (int i = 0; i < HIF_BENCH_ROWS; i++)
        wrong += hif_predict(hif_bench_features[i]) != hif_bench_pred[i];
    bench_mark(MARK_END);

    bench_mark(MARK_PROFILE);
    hif_extract_features_marked(hif_tv_x[0], hif_tv_phase[0], feats[0], profile_mark);

    uart_puts("BENCH,");
    uart_putu(BENCH_EXTRACT_REPEAT * HIF_TV_COUNT);
    uart_putc(',');
    uart_putu(HIF_BENCH_ROWS);
    uart_putc(',');
    uart_putu(wrong);
    uart_puts("\r\n");
    /* The benchmarked windows, through the whole pipeline, for the usual checks. */
    for (int k = 0; k < HIF_TV_COUNT; k++) {
        print_features(k, feats[k]);
        print_prediction(k, hif_predict(feats[k]));
    }
}
#endif

int main(void)
{
    uart_init();

#if defined(HIF_MODE_MODEL) || defined(HIF_EXTRACT)
    if (hif_model_n_features != HIF_N_FEATURES) {
        uart_puts("CONFIG MISMATCH: model.h and test_vectors.h\r\n");
        for (;;)
            ;
    }
#endif

#if defined(HIF_MODE_MODEL)
    run_model();
#elif defined(HIF_MODE_FEATURES)
    run_extract(0);
#elif defined(HIF_MODE_FULL)
    run_extract(1);
#elif defined(HIF_MODE_BENCH)
    run_bench();
#else
    uart_puts("HELLO\r\n");
    /* One hardware float operation, so a missing or disabled FPU shows up
     * here as HARDFAULT instead of later inside the feature code. volatile
     * stops the compiler from folding it at build time. */
    volatile float a = 1.5f, b = 2.25f;
    uart_puts(a * b == 3.375f ? "FPU OK\r\n" : "FPU WRONG\r\n");
#endif

    uart_puts("STACK,");
    uart_putu(stack_peak_bytes());
    uart_puts("\r\nDONE\r\n");
    for (;;)
        ;
}
