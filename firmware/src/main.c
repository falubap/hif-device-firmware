/* HIF region locator firmware.
 *
 * Build modes (make MODE=...):
 *   hello   boot check: HELLO, one hardware float op, FPU OK
 *   model   Step 3: predict from the Python reference features in test_vectors.h, so the model
 *           is checked on the target before feature extraction exists
 *
 * Output lines are parsed by python/run_device.py:
 *   P,<test row>,<predicted region>
 *   DONE
 */

#include "uart.h"

#if defined(HIF_MODE_MODEL)
#include "model.h"
#include "test_vectors.h"

_Static_assert(HIF_TV_N_FEATURES == HIF_N_FEATURES, "test_vectors.h and model.h disagree");

static void run_model(void)
{
    for (int k = 0; k < HIF_TV_COUNT; k++) {
        int pred = hif_model_predict(hif_tv_features[k]);
        uart_puts("P,");
        uart_putu(hif_tv_index[k]);
        uart_putc(',');
        uart_putu((uint32_t)pred);
        uart_puts("\r\n");
    }
}
#endif

int main(void)
{
    uart_init();

#if defined(HIF_MODE_MODEL)
    run_model();
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
