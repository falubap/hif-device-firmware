/* Host-side check of the generated model.h (Step 3, Gate 3).
 *
 * Reads float32 feature vectors (HIF_N_FEATURES each) from stdin and prints the predicted
 * region, one per line. python/convert.py builds this with the host compiler and compares the
 * output with scikit-learn on the whole test split. */

#include <stdio.h>

#include "model.h"

int main(void)
{
    float features[HIF_N_FEATURES];
    while (fread(features, sizeof features, 1, stdin) == 1)
        printf("%d\n", hif_model_predict(features));
    return 0;
}
