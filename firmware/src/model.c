/* The generated model in its own translation unit, so the map file attributes its flash and RAM
 * to it (static inline code in main.c would be folded into main) and inference can be timed
 * as one call. */

#include "model.h"
#include "model_api.h"

const int hif_model_n_features = HIF_N_FEATURES;

__attribute__((noinline)) int hif_predict(const float *features)
{
    return hif_model_predict(features);
}
