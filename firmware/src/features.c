/* Feature extraction. Reference: python/features.py (definitions), python/fast.py (algorithms).
 *
 * Two phases per window:
 *   1. coefficients: each transform fills its slot caches (one slot per derived channel the
 *      configuration uses), computing only the orders/bands some feature reads;
 *   2. features: per feature, the parameter series over the 10 cycles of both windows, then the
 *      statistic.
 * All buffers are static and sized by the generated feature_config.h. */

#include <math.h>
#include <string.h>

#include "dsp2560.h"
#include "hif_features.h"
#include "feature_config.h"

#define EPS 1e-12f
#define PI_F 3.14159265358979323846f
#define N_SAMPLES HIF_SAMPLES

typedef struct {
    float re, im;
} cpx;

/* ------------------------------------------------------------------ caches and scratch */

#if HIF_CFG_DFT_SLOTS
static cpx dft_coef[HIF_CFG_DFT_SLOTS][2][HIF_ORDERS][HIF_CYCLES];
#endif
#if HIF_CFG_STOCKWELL_SLOTS
static cpx st_coef[HIF_CFG_STOCKWELL_SLOTS][2][HIF_ORDERS][HIF_CYCLES];
/* Voice h sums |m| <= 15h around k0 = 10h (python: fast.STOCKWELL_SPAN), so it reads bins up
 * to 25h. Gaussians are tabulated for m >= 0 (they are even); B[m] for m >= 0 (B[-m] = B[m]*). */
#define ST_M(h) (15 * (h))
static float st_gauss[HIF_CFG_STOCKWELL_GAUSS_LEN];
static uint16_t st_gauss_offset[HIF_ORDERS];
static cpx st_kernel[ST_M(HIF_CFG_STOCKWELL_MAX_ORDER) + 1];
#endif
#if HIF_CFG_STOCKWELL_SLOTS || HIF_CFG_DFT_SLOTS
static cpx raw_coef[HIF_ORDERS][HIF_CYCLES];    /* one raw signal's transform */
#endif
#if HIF_CFG_NEED_HILBERT
#define SPECTRUM_BINS (N_SAMPLES / 2 + 1)
#elif HIF_CFG_STOCKWELL_SLOTS
#define SPECTRUM_BINS (25 * HIF_CFG_STOCKWELL_MAX_ORDER + 1)
#endif
#ifdef SPECTRUM_BINS
static float spectrum[2 * SPECTRUM_BINS];
#endif
#if HIF_CFG_WAVELET_SLOTS
static float wav_rms[HIF_CFG_WAVELET_SLOTS][2][HIF_BANDS][HIF_CYCLES];
static float wav_in[N_SAMPLES];                  /* derived channel waveform */
static float wav_ping[N_SAMPLES], wav_pong[N_SAMPLES];
#endif

/* ------------------------------------------------------------------ init */

void hif_features_init(void)
{
    dsp2560_init();
#if HIF_CFG_STOCKWELL_SLOTS
    /* Gaussian of each needed voice, exp(-2 pi^2 m^2 / k0^2), m = 0..15h (in double: the
     * argument reaches 44, where a float32 exponent loses ~3e-6). */
    uint32_t orders = 0;
    for (int s = 0; s < HIF_CFG_STOCKWELL_SLOTS; s++)
        orders |= hif_cfg_stockwell_orders[s];
    uint16_t offset = 0;
    for (int h = 1; h <= HIF_ORDERS; h++) {
        if (!(orders & (1u << (h - 1))))
            continue;
        st_gauss_offset[h - 1] = offset;
        double k0 = 10.0 * h;
        for (int m = 0; m <= ST_M(h); m++)
            st_gauss[offset++] = (float)exp(-2.0 * 9.8696044010893586 * m * m / (k0 * k0));
    }
    /* B[m] = (1/256) sum_{u<256} e^{j2pi mu/N} = (1 - e^{j2pi m/10}) / (256 (1 - e^{j2pi m/N})) */
    st_kernel[0].re = 1.0f;
    st_kernel[0].im = 0.0f;
    for (int m = 1; m <= ST_M(HIF_CFG_STOCKWELL_MAX_ORDER); m++) {
        const double two_pi = 6.283185307179586477;
        double nr = 1.0 - cos(two_pi * m / 10.0), ni = -sin(two_pi * m / 10.0);
        double dr = 256.0 * (1.0 - cos(two_pi * m / N_SAMPLES)), di = -256.0 * sin(two_pi * m / N_SAMPLES);
        double den = dr * dr + di * di;
        st_kernel[m].re = (float)((nr * dr + ni * di) / den);
        st_kernel[m].im = (float)((ni * dr - nr * di) / den);
    }
#endif
}

/* ------------------------------------------------------------------ channels */

#if HIF_CFG_DFT_SLOTS || HIF_CFG_STOCKWELL_SLOTS
/* Weight of raw signal `raw` in derived channel `ch` (python: features.channel_phasors).
 * Returns 0 if the raw signal does not contribute. */
static int channel_weight(int ch, int raw, int phase, cpx *w)
{
    /* Negative sequence: (A + a^2 B + a C) / 3, a = e^{j 2pi/3}. */
    static const cpx negseq[3] = {
        {1.0f / 3.0f, 0.0f}, {-1.0f / 6.0f, -0.288675135f}, {-1.0f / 6.0f, 0.288675135f},
    };
    w->re = 1.0f;
    w->im = 0.0f;
    switch (ch) {
    case HIF_CH_FAULT_V:   return raw == HIF_VA + phase;
    case HIF_CH_FAULT_I:   return raw == HIF_IA + phase;
    case HIF_CH_NEUTRAL_I: return raw == HIF_IN;
    case HIF_CH_VSUM:      return raw <= HIF_VC;
    case HIF_CH_NEGSEQ_V:
    case HIF_CH_NEGSEQ_I: {
        int k = raw - (ch == HIF_CH_NEGSEQ_V ? HIF_VA : HIF_IA);
        if (k < 0 || k > 2)
            return 0;
        *w = negseq[k];
        return 1;
    }
    }
    return 0;
}
#endif

/* ------------------------------------------------------------------ transforms */

#if HIF_CFG_DFT_SLOTS
/* Per-cycle DFT, harmonics in `orders`, scaled to peak amplitude (python: features.dft). */
static void dft_cycles(const float *x, uint32_t orders, cpx out[HIF_ORDERS][HIF_CYCLES])
{
    float spec[DSP_SPC];
    for (int c = 0; c < HIF_CYCLES; c++) {
        dsp_rfft256(x + DSP_SPC * c, spec);
        for (int h = 1; h <= HIF_ORDERS; h++) {
            if (orders & (1u << (h - 1))) {
                out[h - 1][c].re = spec[2 * h] * (2.0f / DSP_SPC);
                out[h - 1][c].im = spec[2 * h + 1] * (2.0f / DSP_SPC);
            }
        }
    }
}
#endif

#if HIF_CFG_STOCKWELL_SLOTS
/* Per-cycle means of the S-transform voices in `orders` (python: fast.stockwell):
 *   X_{h,c} = (2/N) sum_{|m|<=15h} H[10h+m] G_h[m] B[m] e^{j2pi mc/10},
 * with the sum over m grouped by m mod 10 into S_r, then a 10-point DFT over cycles. */
static void stockwell_voices(const float *x, uint32_t orders, cpx out[HIF_ORDERS][HIF_CYCLES])
{
    int h_max = 32 - __builtin_clz(orders);
    dsp2560_dft(x, 25 * h_max, spectrum);
    for (int h = 1; h <= h_max; h++) {
        if (!(orders & (1u << (h - 1))))
            continue;
        const int k0 = 10 * h, M = ST_M(h);
        const float *g = st_gauss + st_gauss_offset[h - 1];
        cpx S[10] = {{0}};
        for (int m = -M; m <= M; m++) {
            int k = k0 + m, am = m < 0 ? -m : m;
            float h_re = spectrum[2 * (k < 0 ? -k : k)];
            float h_im = k < 0 ? -spectrum[-2 * k + 1] : spectrum[2 * k + 1];
            float b_re = st_kernel[am].re, b_im = m < 0 ? -st_kernel[am].im : st_kernel[am].im;
            float v_re = (h_re * b_re - h_im * b_im) * g[am];
            float v_im = (h_re * b_im + h_im * b_re) * g[am];
            int r = ((m % 10) + 10) % 10;
            S[r].re += v_re;
            S[r].im += v_im;
        }
        for (int c = 0; c < HIF_CYCLES; c++) {
            float acc_re = 0.0f, acc_im = 0.0f;
            for (int r = 0; r < 10; r++) {
                float w_re, w_im;
                dsp_twiddle(-DSP_SPC * ((r * c) % 10), &w_re, &w_im);   /* e^{+j2pi rc/10} */
                acc_re += S[r].re * w_re - S[r].im * w_im;
                acc_im += S[r].re * w_im + S[r].im * w_re;
            }
            out[h - 1][c].re = acc_re * (2.0f / N_SAMPLES);
            out[h - 1][c].im = acc_im * (2.0f / N_SAMPLES);
        }
    }
}
#endif

#if HIF_CFG_DFT_SLOTS || HIF_CFG_STOCKWELL_SLOTS
/* Fill every slot of one analytic transform for window w: transform each contributing raw
 * signal once, and add it with its weight into the slots that use it. */
static void analytic(int transform, const float x[HIF_RAW][HIF_SAMPLES], int phase, int n_slots,
                     const uint8_t *channels, const uint32_t *orders,
                     cpx coef[][2][HIF_ORDERS][HIF_CYCLES], int w)
{
    for (int s = 0; s < n_slots; s++)
        memset(coef[s][w], 0, sizeof coef[s][w]);

    for (int raw = 0; raw < HIF_RAW; raw++) {
        uint32_t need = 0;
        cpx wt;
        for (int s = 0; s < n_slots; s++)
            if (channel_weight(channels[s], raw, phase, &wt))
                need |= orders[s];
        if (!need)
            continue;
#if HIF_CFG_DFT_SLOTS
        if (transform == HIF_DFT)
            dft_cycles(x[raw], need, raw_coef);
#endif
#if HIF_CFG_STOCKWELL_SLOTS
        if (transform == HIF_STOCKWELL)
            stockwell_voices(x[raw], need, raw_coef);
#endif
        for (int s = 0; s < n_slots; s++) {
            if (!channel_weight(channels[s], raw, phase, &wt))
                continue;
            for (int h = 0; h < HIF_ORDERS; h++) {
                if (!(orders[s] & (1u << h)))
                    continue;
                for (int c = 0; c < HIF_CYCLES; c++) {
                    cpx v = raw_coef[h][c];
                    coef[s][w][h][c].re += wt.re * v.re - wt.im * v.im;
                    coef[s][w][h][c].im += wt.re * v.im + wt.im * v.re;
                }
            }
        }
    }
}
#endif

#if HIF_CFG_WAVELET_SLOTS
/* db4 decomposition filters / sqrt(2) (pywt.swt norm=True). */
static const float swt_lo[8] = {
    -7.493494665e-03f, 2.325180054e-02f, 2.180815024e-02f, -1.322535837e-01f,
    -1.978751312e-02f, 4.461000691e-01f, 5.054728575e-01f, 1.629017140e-01f,
};
static const float swt_hi[8] = {
    -1.629017140e-01f, 5.054728575e-01f, -4.461000691e-01f, -1.978751312e-02f,
    1.322535837e-01f, 2.180815024e-02f, -2.325180054e-02f, -7.493494665e-03f,
};

#if HIF_CFG_NEED_HILBERT
/* acc += cos(t) x - sin(t) H{x}, H{x} = irfft of -j X on 0 < k < N/2 (python: fast.hilbert_rotate).
 * Uses wav_ping as scratch. */
static void add_rotated(const float *x, float cos_t, float sin_t, float *acc)
{
    dsp2560_dft(x, N_SAMPLES / 2, spectrum);
    for (int k = 0; k <= N_SAMPLES / 2; k++) {
        float re = spectrum[2 * k], im = spectrum[2 * k + 1];
        int edge = k == 0 || k == N_SAMPLES / 2;
        spectrum[2 * k] = edge ? 0.0f : im;        /* -j (re + j im) = im - j re */
        spectrum[2 * k + 1] = edge ? 0.0f : -re;
    }
    dsp2560_irfft(spectrum, wav_ping);
    for (int i = 0; i < N_SAMPLES; i++)
        acc[i] += cos_t * x[i] - sin_t * wav_ping[i];
}
#endif

/* Derived channel waveform (python: features.channel_waveform). Returns a pointer to it. */
static const float *waveform(int ch, const float x[HIF_RAW][HIF_SAMPLES], int phase)
{
    switch (ch) {
    case HIF_CH_FAULT_V:   return x[HIF_VA + phase];
    case HIF_CH_FAULT_I:   return x[HIF_IA + phase];
    case HIF_CH_NEUTRAL_I: return x[HIF_IN];
    case HIF_CH_VSUM:
        for (int i = 0; i < N_SAMPLES; i++)
            wav_in[i] = x[HIF_VA][i] + x[HIF_VB][i] + x[HIF_VC][i];
        return wav_in;
#if HIF_CFG_NEED_HILBERT
    case HIF_CH_NEGSEQ_V:
    case HIF_CH_NEGSEQ_I: {
        /* (a + rot(b, 4pi/3) + rot(c, 2pi/3)) / 3 */
        const float (*abc)[HIF_SAMPLES] = x + (ch == HIF_CH_NEGSEQ_V ? HIF_VA : HIF_IA);
        memcpy(wav_in, abc[0], sizeof wav_in);
        add_rotated(abc[1], -0.5f, -0.866025404f, wav_in);
        add_rotated(abc[2], -0.5f, 0.866025404f, wav_in);
        for (int i = 0; i < N_SAMPLES; i++)
            wav_in[i] /= 3.0f;
        return wav_in;
    }
#endif
    }
    return x[HIF_IN];
}

/* Per-cycle RMS of the SWT bands in `bands` (bit j: d_{j+1}, bit 8: a8), python: features.wavelet.
 * Level j filters the previous approximation with the filters upsampled by 2^j, circularly:
 *   out[i] = sum_k f[k] x[(i - 2^j (k - 4)) mod N]. */
static void wavelet_bands(const float *in, uint16_t bands, float rms[HIF_BANDS][HIF_CYCLES])
{
    int levels = (bands & (1u << 8)) ? 8 : 16 - __builtin_clz((uint32_t)bands << 16);
    const float *a = in;
    for (int level = 0; level < levels; level++) {
        const int step = 1 << level;
        const int need_d = bands & (1u << level);
        float *out = (level & 1) ? wav_pong : wav_ping;
        float energy[HIF_CYCLES] = {0};
        for (int i = 0; i < N_SAMPLES; i++) {
            float d = 0.0f, ap = 0.0f;
            for (int k = 0; k < 8; k++) {
                int idx = i - step * (k - 4);
                if (idx < 0)
                    idx += N_SAMPLES;
                else if (idx >= N_SAMPLES)
                    idx -= N_SAMPLES;
                d += swt_hi[k] * a[idx];
                ap += swt_lo[k] * a[idx];
            }
            out[i] = ap;
            if (need_d)
                energy[i / DSP_SPC] += d * d;
        }
        if (need_d)
            for (int c = 0; c < HIF_CYCLES; c++)
                rms[level][c] = sqrtf(energy[c] / DSP_SPC);
        a = out;
    }
    if (bands & (1u << 8)) {
        for (int c = 0; c < HIF_CYCLES; c++) {
            float e = 0.0f;
            for (int i = 0; i < DSP_SPC; i++)
                e += a[DSP_SPC * c + i] * a[DSP_SPC * c + i];
            rms[8][c] = sqrtf(e / DSP_SPC);
        }
    }
}
#endif

/* ------------------------------------------------------------------ parameters */

static float cabs_(cpx v) { return sqrtf(v.re * v.re + v.im * v.im); }

/* The parameter of feature f over the 10 cycles of window w (python: features.parameter). */
static void parameter_series(const hif_feature_t *f, int w, float out[HIF_CYCLES])
{
#if HIF_CFG_WAVELET_SLOTS
    if (f->transform == HIF_WAVELET) {
        const float *a = wav_rms[f->slot][w][f->row];
        for (int c = 0; c < HIF_CYCLES; c++) {
            if (f->param == HIF_ENERGY)
                out[c] = a[c] * a[c];
            else if (f->param == HIF_IMPEDANCE)
                out[c] = a[c] / fmaxf(wav_rms[f->slot2][w][f->row][c], EPS);
            else
                out[c] = a[c];
        }
        return;
    }
#endif
    cpx (*coef)[2][HIF_ORDERS][HIF_CYCLES] = 0;
#if HIF_CFG_DFT_SLOTS
    if (f->transform == HIF_DFT)
        coef = dft_coef;
#endif
#if HIF_CFG_STOCKWELL_SLOTS
    if (f->transform == HIF_STOCKWELL)
        coef = st_coef;
#endif
    if (!coef)
        return;
    const cpx *a = coef[f->slot][w][f->row];
    const cpx *b = f->slot2 != 0xFF ? coef[f->slot2][w][f->row] : 0;
    for (int c = 0; c < HIF_CYCLES; c++) {
        switch (f->param) {
        case HIF_MODULE:
            out[c] = cabs_(a[c]);
            break;
        case HIF_ENERGY:
            out[c] = a[c].re * a[c].re + a[c].im * a[c].im;
            break;
        case HIF_ANGLE:
            out[c] = atan2f(a[c].im, a[c].re);
            break;
        case HIF_THD: {
            float sum = 0.0f;
            for (int h = 1; h < HIF_ORDERS; h++) {
                cpx v = coef[f->slot][w][h][c];
                sum += v.re * v.re + v.im * v.im;
            }
            out[c] = sqrtf(sum) / fmaxf(cabs_(coef[f->slot][w][0][c]), EPS);
            break;
        }
        case HIF_IMPEDANCE:
            out[c] = cabs_(a[c]) / fmaxf(cabs_(b[c]), EPS);
            break;
        case HIF_VI_ANGLE:   /* angle(a conj(b)) */
            out[c] = atan2f(a[c].im * b[c].re - a[c].re * b[c].im,
                            a[c].re * b[c].re + a[c].im * b[c].im);
            break;
        }
    }
}

/* ------------------------------------------------------------------ statistics */

static float mean10(const float *v)
{
    float s = 0.0f;
    for (int c = 0; c < HIF_CYCLES; c++)
        s += v[c];
    return s / HIF_CYCLES;
}

/* Circular mean in [-pi, pi) (python: features.circmean). */
static float circmean10(const float *v)
{
    float s = 0.0f, co = 0.0f;
    for (int c = 0; c < HIF_CYCLES; c++) {
        s += sinf(v[c]);
        co += cosf(v[c]);
    }
    float m = atan2f(s / HIF_CYCLES, co / HIF_CYCLES);
    return m >= PI_F ? m - 2.0f * PI_F : m;
}

/* python: features.statistic */
static float statistic(int stat, int circular, const float *pre, const float *post)
{
    if (circular) {
        if (stat == HIF_MEAN)
            return circmean10(post);
        if (stat == HIF_STD) {
            /* sqrt(-2 ln R), R = |mean e^{j theta}|. For a steady phase R is 1 - 1e-7, which
             * float32 cannot resolve, so 1 - R is formed without cancellation: rotate by the
             * circular mean, then mean cos = 1 - u with u = mean 2 sin^2(d/2), and
             * 1 - R = (1 - R^2) / (1 + R) = (2u - u^2 - s^2) / (1 + R). */
            float mu = circmean10(post), u = 0.0f, s = 0.0f;
            for (int c = 0; c < HIF_CYCLES; c++) {
                float d = post[c] - mu, h = sinf(0.5f * d);
                u += 2.0f * h * h;
                s += sinf(d);
            }
            u /= HIF_CYCLES;
            s /= HIF_CYCLES;
            float r = sqrtf((1.0f - u) * (1.0f - u) + s * s);
            float one_minus_r = fmaxf((2.0f * u - u * u - s * s) / (1.0f + r), 0.0f);
            return sqrtf(-2.0f * log1pf(-one_minus_r));
        }
        float d = circmean10(post) - circmean10(pre);   /* HIF_DELTA, wrapped */
        return atan2f(sinf(d), cosf(d));
    }

    float mean = mean10(post), lo = post[0], hi = post[0];
    for (int c = 1; c < HIF_CYCLES; c++) {
        lo = fminf(lo, post[c]);
        hi = fmaxf(hi, post[c]);
    }
    switch (stat) {
    case HIF_MEAN:
        return mean;
    case HIF_MAX:
        return hi;
    case HIF_AMPLITUDE:
        return hi - lo;
    case HIF_STD:
    case HIF_KURTOSIS: {
        float m2 = 0.0f, m4 = 0.0f;
        for (int c = 0; c < HIF_CYCLES; c++) {
            float d = post[c] - mean;
            m2 += d * d;
            m4 += d * d * d * d;
        }
        m2 /= HIF_CYCLES;
        m4 /= HIF_CYCLES;
        if (stat == HIF_STD)
            return sqrtf(m2);
        /* Unbiased Fisher kurtosis; 0 when the variance is at the mean's resolution (scipy). */
        float res = 1e-15f * mean;
        if (m2 <= res * res)
            return 0.0f;
        const float n = HIF_CYCLES;
        return ((n * n - 1.0f) * m4 / (m2 * m2) - 3.0f * (n - 1.0f) * (n - 1.0f))
               / ((n - 2.0f) * (n - 3.0f));
    }
    case HIF_ENTROPY: {
        float total = 0.0f, e = 0.0f;
        for (int c = 0; c < HIF_CYCLES; c++)
            total += fabsf(post[c]);
        total = fmaxf(total, EPS);
        for (int c = 0; c < HIF_CYCLES; c++) {
            float p = fabsf(post[c]) / total;
            e -= p * logf(fmaxf(p, EPS));
        }
        return e;
    }
    case HIF_DELTA:
        return mean - mean10(pre);
    case HIF_RATIO: {
        float p = mean10(pre);
        float guard = fabsf(p) < EPS ? copysignf(EPS, p + EPS) : p;
        return mean / guard;
    }
    }
    return 0.0f;
}

/* ------------------------------------------------------------------ extraction */

void hif_extract_features(const float x[2][HIF_RAW][HIF_SAMPLES], int phase, float *out)
{
    for (int w = 0; w < 2; w++) {
#if HIF_CFG_DFT_SLOTS
        analytic(HIF_DFT, x[w], phase, HIF_CFG_DFT_SLOTS, hif_cfg_dft_channel,
                 hif_cfg_dft_orders, dft_coef, w);
#endif
#if HIF_CFG_STOCKWELL_SLOTS
        analytic(HIF_STOCKWELL, x[w], phase, HIF_CFG_STOCKWELL_SLOTS, hif_cfg_stockwell_channel,
                 hif_cfg_stockwell_orders, st_coef, w);
#endif
#if HIF_CFG_WAVELET_SLOTS
        for (int s = 0; s < HIF_CFG_WAVELET_SLOTS; s++)
            wavelet_bands(waveform(hif_cfg_wavelet_channel[s], x[w], phase),
                          hif_cfg_wavelet_bands[s], wav_rms[s][w]);
#endif
    }

    for (int i = 0; i < HIF_CFG_N_FEATURES; i++) {
        const hif_feature_t *f = &hif_cfg_features[i];
        float pre[HIF_CYCLES], post[HIF_CYCLES];
        parameter_series(f, HIF_PRE, pre);
        parameter_series(f, HIF_POST, post);
        int circular = f->param == HIF_ANGLE || f->param == HIF_VI_ANGLE;
        out[i] = statistic(f->stat, circular, pre, post);
    }
}
