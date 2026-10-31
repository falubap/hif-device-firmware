/* Host-side run of the C feature extractor (Step 4): the same features.c / dsp2560.c and
 * CMSIS-DSP sources, compiled for the PC, so parity with python/features.py can be checked on
 * every test window in seconds. The device run (python/run_device.py features) is the real check.
 *
 * stdin: per window, int32 test row, int32 faulted phase, then 2 x 7 x 2560 float32.
 * stdout: F,<row>,<feature bits hex>,... as the firmware prints it. */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "hif_features.h"
#include "feature_config.h"

static float x[2][HIF_RAW][HIF_SAMPLES];

int main(void)
{
    int32_t head[2];
    float feats[HIF_CFG_N_FEATURES];
    hif_features_init();
    while (fread(head, sizeof head, 1, stdin) == 1 && fread(x, sizeof x, 1, stdin) == 1) {
        hif_extract_features(x, head[1], feats);
        printf("F,%d", head[0]);
        for (int i = 0; i < HIF_CFG_N_FEATURES; i++) {
            uint32_t bits;
            memcpy(&bits, &feats[i], sizeof bits);
            printf(",%08x", bits);
        }
        printf("\n");
    }
    return 0;
}
