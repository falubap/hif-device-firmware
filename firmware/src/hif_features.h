#ifndef HIF_FEATURES_H
#define HIF_FEATURES_H

/* Feature extraction: the thesis's feature contract in C. Reference: python/features.py (the
 * definitions) and python/fast.py (the algorithms used here).
 *
 * Which features are computed is set by the generated include/feature_config.h: a table of
 * hif_feature_t plus, per transform, the derived channels ("slots") and orders the table needs.
 * Only those are computed, and buffers are sized by them. */

#include <stdint.h>

/* Places a large buffer in the 64 KB CCM (linker.ld) instead of the 128 KB main SRAM. */
#define HIF_CCM __attribute__((section(".ccmram")))

#define HIF_SAMPLES   2560      /* 10 cycles x 256 samples, fs = 15360 Hz */
#define HIF_RAW       7         /* Va Vb Vc Ia Ib Ic In */
#define HIF_CYCLES    10
#define HIF_ORDERS    20        /* harmonics 1..20 */
#define HIF_BANDS     9         /* wavelet d1..d8, a8 */

enum { HIF_PRE, HIF_POST };
enum { HIF_VA, HIF_VB, HIF_VC, HIF_IA, HIF_IB, HIF_IC, HIF_IN };
enum hif_transform { HIF_DFT, HIF_STOCKWELL, HIF_WAVELET };
enum hif_channel {
    HIF_CH_FAULT_V, HIF_CH_FAULT_I, HIF_CH_NEGSEQ_V, HIF_CH_NEGSEQ_I, HIF_CH_NEUTRAL_I, HIF_CH_VSUM,
};
enum hif_param { HIF_MODULE, HIF_ENERGY, HIF_ANGLE, HIF_THD, HIF_IMPEDANCE, HIF_VI_ANGLE };
enum hif_stat {
    HIF_MEAN, HIF_MAX, HIF_AMPLITUDE, HIF_STD, HIF_KURTOSIS, HIF_ENTROPY, HIF_DELTA, HIF_RATIO,
};

typedef struct {
    uint8_t transform;
    uint8_t slot;       /* channel slot of that transform */
    uint8_t slot2;      /* second slot of a V/I pair, else 0xFF */
    uint8_t row;        /* harmonic - 1, or band (d1..d8 = 0..7, a8 = 8); 0 for THD */
    uint8_t param;
    uint8_t stat;
} hif_feature_t;

void hif_features_init(void);

/* All configured features of one window; out has HIF_N_FEATURES entries, model input order.
 * phase: faulted phase 0..2 (A..C). */
void hif_extract_features(const float x[2][HIF_RAW][HIF_SAMPLES], int phase, float *out);

/* Same, calling mark() after each stage (DFT, Stockwell, wavelet, statistics) so a benchmark
 * can split the cost; stages that the configuration does not use still call it. */
void hif_extract_features_marked(const float x[2][HIF_RAW][HIF_SAMPLES], int phase, float *out,
                                 void (*mark)(void));

#endif
