#ifndef DSP2560_H
#define DSP2560_H

/* 2560-point real transforms (10 cycles x 256 samples) built from CMSIS-DSP 256-point FFTs by
 * decimation, since CMSIS supports powers of two only. Reference: python/fast.py. */

#include <stdint.h>

#define DSP_N   2560
#define DSP_SPC 256          /* samples per cycle; also the CMSIS FFT size */
#define DSP_D   10           /* DSP_N = DSP_D * DSP_SPC */

void dsp2560_init(void);

/* e^{-j 2 pi q / DSP_N}, for any integer q. */
void dsp_twiddle(int32_t q, float *re, float *im);

/* Real 256-point FFT, CMSIS packing: out[0] = X0, out[1] = X128, out[2k], out[2k+1] = X_k. */
void dsp_rfft256(const float *in, float *out);

/* H[k] for k = 0..k_max (k_max <= DSP_N / 2) of a real 2560-sample signal, interleaved re/im.
 * (python: fast.dft2560) */
void dsp2560_dft(const float *x, int k_max, float *H);

/* Real signal from its bins 0..DSP_N/2 (interleaved re/im). (python: fast.irfft2560) */
void dsp2560_irfft(const float *H, float *x);

#endif
