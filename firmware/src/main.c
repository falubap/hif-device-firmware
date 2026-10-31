/* HIF region locator firmware.
 *
 * Build modes (make MODE=...):
 *   hello     boot check: HELLO, one hardware float op, FPU OK
 *   model     Step 3: predict from the Python reference features in test_vectors.h, so the
 *             model is checked on the target before feature extraction exists
 *   features  Step 4: extract the features of each raw window in test_vectors.h
 *   full      Step 5: raw window -> features -> model, for each window in test_vectors.h
 *
 * Output lines are parsed by python/run_device.py:
 *   P,<test row>,<predicted region>
 *   F,<test row>,<feature 0 bits>,...   (float32 bit patterns, hex)
 *   DONE
 */

#include <string.h>

#include "uart.h"

#if defined(HIF_MODE_FEATURES) || defined(HIF_MODE_FULL)
#define HIF_EXTRACT
#endif

#if defined(HIF_MODE_MODEL) || defined(HIF_EXTRACT)
#include "model.h"
#include "test_vectors.h"

_Static_assert(HIF_TV_N_FEATURES == HIF_N_FEATURES, "test_vectors.h and model.h disagree");

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

_Static_assert(HIF_CFG_N_FEATURES == HIF_N_FEATURES, "feature_config.h and model.h disagree");
_Static_assert(HIF_TV_HAS_RAW, "features mode needs raw windows (export_vectors.py --batch)");
#endif

#if defined(HIF_MODE_MODEL)
static void run_model(void)
{
    for (int k = 0; k < HIF_TV_COUNT; k++)
        print_prediction(k, hif_model_predict(hif_tv_features[k]));
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

/* Features of every window; with predict, also the model's region (full pipeline). */
static void run_extract(int predict)
{
    static float feats[HIF_N_FEATURES];
    hif_features_init();
    for (int k = 0; k < HIF_TV_COUNT; k++) {
        hif_extract_features(hif_tv_x[k], hif_tv_phase[k], feats);
        print_features(k, feats);
        if (predict)
            print_prediction(k, hif_model_predict(feats));
    }
}
#endif

int main(void)
{
    uart_init();

#if defined(HIF_MODE_MODEL)
    run_model();
#elif defined(HIF_MODE_FEATURES)
    run_extract(0);
#elif defined(HIF_MODE_FULL)
    run_extract(1);
#else
    uart_puts("HELLO\r\n");
    /* One hardware float operation, so a missing or disabled FPU shows up
     * here as HARDFAULT instead of later inside the feature code. volatile
     * stops the compiler from folding it at build time. */
    volatile float a = 1.5f, b = 2.25f;
    uart_puts(a * b == 3.375f ? "FPU OK\r\n" : "FPU WRONG\r\n");
#endif

    uart_puts("DONE\r\n");
    for (;;)
        ;
}
