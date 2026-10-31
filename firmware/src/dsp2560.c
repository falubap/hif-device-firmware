/* 2560-point transforms from CMSIS 256-point FFTs. Reference: python/fast.py. */

#include <math.h>
#include <string.h>

#include "arm_math.h"
#include "dsp2560.h"

/* cos(2 pi q / DSP_N) for q = 0..DSP_N/4; every other twiddle follows by symmetry. */
static float cos_quarter[DSP_N / 4 + 1];
static arm_rfft_fast_instance_f32 rfft256;

/* Ten decimated spectra Y_s (CMSIS packing), shared by both directions. */
static float spectra[DSP_D][DSP_SPC];
static float scratch[DSP_SPC];

void dsp2560_init(void)
{
    for (int q = 0; q <= DSP_N / 4; q++)
        cos_quarter[q] = cosf(6.28318530717958647692f * (float)q / DSP_N);
    arm_rfft_fast_init_256_f32(&rfft256);
}

/* cos(2 pi q / N) for q in [0, N). */
static float cos_q(int32_t q)
{
    if (q > DSP_N / 2)
        q = DSP_N - q;
    return q <= DSP_N / 4 ? cos_quarter[q] : -cos_quarter[DSP_N / 2 - q];
}

void dsp_twiddle(int32_t q, float *re, float *im)
{
    q %= DSP_N;
    if (q < 0)
        q += DSP_N;
    *re = cos_q(q);
    /* sin(2 pi q / N) = cos(2 pi (q - N/4) / N); the twiddle is e^{-j...}. */
    *im = -cos_q((q + DSP_N - DSP_N / 4) % DSP_N);
}

void dsp_rfft256(const float *in, float *out)
{
    memcpy(scratch, in, sizeof scratch);   /* CMSIS overwrites its input */
    arm_rfft_fast_f32(&rfft256, scratch, out, 0);
}

/* Bin k (0..255) of a packed 256-point real spectrum. */
static void packed_bin(const float *Y, int k, float *re, float *im)
{
    if (k == 0) {
        *re = Y[0];
        *im = 0.0f;
    } else if (k == DSP_SPC / 2) {
        *re = Y[1];
        *im = 0.0f;
    } else if (k < DSP_SPC / 2) {
        *re = Y[2 * k];
        *im = Y[2 * k + 1];
    } else {
        *re = Y[2 * (DSP_SPC - k)];
        *im = -Y[2 * (DSP_SPC - k) + 1];
    }
}

void dsp2560_dft(const float *x, int k_max, float *H)
{
    float column[DSP_SPC];
    for (int s = 0; s < DSP_D; s++) {
        for (int r = 0; r < DSP_SPC; r++)
            column[r] = x[DSP_D * r + s];
        dsp_rfft256(column, spectra[s]);
    }
    /* H[k] = sum_s e^{-j2pi sk/N} Y_s[k mod 256] */
    for (int k = 0; k <= k_max; k++) {
        float acc_re = 0.0f, acc_im = 0.0f;
        for (int s = 0; s < DSP_D; s++) {
            float y_re, y_im, w_re, w_im;
            packed_bin(spectra[s], k % DSP_SPC, &y_re, &y_im);
            dsp_twiddle(s * k, &w_re, &w_im);
            acc_re += w_re * y_re - w_im * y_im;
            acc_im += w_re * y_im + w_im * y_re;
        }
        H[2 * k] = acc_re;
        H[2 * k + 1] = acc_im;
    }
}

/* Bin k (0..N-1) of a real signal's spectrum given bins 0..N/2. */
static void full_bin(const float *H, int k, float *re, float *im)
{
    if (k <= DSP_N / 2) {
        *re = H[2 * k];
        *im = H[2 * k + 1];
    } else {
        *re = H[2 * (DSP_N - k)];
        *im = -H[2 * (DSP_N - k) + 1];
    }
}

void dsp2560_irfft(const float *H, float *x)
{
    float packed[DSP_SPC];
    for (int s = 0; s < DSP_D; s++) {
        /* Z_s[k1] = sum_{k2} H[k1 + 256 k2] e^{+j2pi (k1 + 256 k2) s / N}, k1 = 0..128. Z_s is
         * Hermitian (x[s::D] is real), so CMSIS's packed half spectrum is enough. */
        for (int k1 = 0; k1 <= DSP_SPC / 2; k1++) {
            float acc_re = 0.0f, acc_im = 0.0f;
            for (int k2 = 0; k2 < DSP_D; k2++) {
                int k = k1 + DSP_SPC * k2;
                float h_re, h_im, w_re, w_im;
                full_bin(H, k, &h_re, &h_im);
                dsp_twiddle(-k * s, &w_re, &w_im);          /* e^{+j...} */
                acc_re += w_re * h_re - w_im * h_im;
                acc_im += w_re * h_im + w_im * h_re;
            }
            if (k1 == 0) {
                packed[0] = acc_re;
            } else if (k1 == DSP_SPC / 2) {
                packed[1] = acc_re;
            } else {
                packed[2 * k1] = acc_re;
                packed[2 * k1 + 1] = acc_im;
            }
        }
        arm_rfft_fast_f32(&rfft256, packed, spectra[s], 1);  /* includes the 1/256 */
        for (int r = 0; r < DSP_SPC; r++)
            x[DSP_D * r + s] = spectra[s][r] / DSP_D;
    }
}
