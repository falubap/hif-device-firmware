# Feature definitions

Exact definitions of the thesis's features as implemented in `python/features.py` (the reference
the firmware is checked against). `features.py` reproduces the thesis's feature matrix to float32
rounding on every test file (`python/export_windows.py`).

## 1. Input

| Item | Value |
| --- | --- |
| Measurement point | substation only (`Msub`) |
| Raw signals | phase voltages $v_A, v_B, v_C$; phase currents $i_A, i_B, i_C$; neutral current $i_N$ |
| Noise | white Gaussian, 40 dB SNR per signal, power taken over the whole record; added before resampling |
| Sampling | cubic resampling onto 256 samples/cycle, $f_s = 60 \cdot 256 = 15360$ Hz |
| Pre-fault window | 10 cycles ending at the fault instant $t_f = 3.780$ s |
| Post-fault window | 10 cycles starting 1 cycle after $t_f$ (the transient cycle is skipped) |
| Window length | $N = 2560$ samples each; cycle $c = 0..9$ covers samples $256c .. 256c+255$ |
| Faulted phase $\varphi$ | from scenario metadata (assumes an upstream phase classifier) |

## 2. Channels

| Channel | Definition |
| --- | --- |
| fault voltage | $v_\varphi$ |
| fault current | $i_\varphi$ |
| neutral current | $i_N$ |
| voltage sum | $v_A + v_B + v_C$ ($= 3v_0$) |
| negative-sequence voltage | $\tfrac13 (V_A + a^2 V_B + a V_C)$, $a = e^{j2\pi/3}$ |
| negative-sequence current | same, with currents |

The negative sequence is formed from complex coefficients, never from real waveforms. DFT and
Stockwell apply it to each harmonic's phasor. The wavelet is real-valued, so there the rotation by
$a$ is applied as a broadband phase shift through the analytic signal:
$\mathrm{Re}\{e^{j\theta}(x + j\mathcal{H}x)\}$.

## 3. Transforms

Each one maps a channel's window to a coefficient matrix $X_{o,c}$: order or band $o$ by cycle $c$.

**DFT** (per cycle, harmonics $h = 1..20$, scaled to peak amplitude):

$$X_{h,c} = \frac{2}{256} \sum_{n=0}^{255} x[256c+n]\, e^{-j2\pi h n/256}$$

**Stockwell** (over the whole 2560-sample window, evaluated only at the harmonic frequencies, then
averaged per cycle). With $H[k]$ the 2560-point DFT of the window and $k_0 = 10h$, the frequency
index of harmonic $h$:

$$S_h[t] = \frac{2}{N} \sum_{m=-N/2+1}^{N/2} H[(m + k_0) \bmod N]\; e^{-2\pi^2 m^2 / k_0^2}\; e^{j2\pi m t / N},
\qquad X_{h,c} = \frac{1}{256} \sum_{t=256c}^{256c+255} S_h[t]$$

The Gaussian's width scales with frequency: low harmonics are smoothed more in time and resolved
more finely in frequency. The mean is taken over the complex voice, so phase is kept.

**Wavelet** (stationary/undecimated db4, 8 levels, periodic extension, filters scaled by
$1/\sqrt2$). Level $j$ filters the previous approximation with the db4 filters upsampled by
$2^{j-1}$. It gives detail bands $d_1..d_8$ and the approximation $a_8$, each 2560 samples long.
Per cycle the coefficient is the band's RMS (real, no phase):

$$X_{b,c} = \sqrt{\tfrac{1}{256} \textstyle\sum_{t \in c} w_b[t]^2}$$

| Band | d1 | d2 | d3 | d4 | d5 | d6 | d7 | d8 | a8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hz | 3840–7680 | 1920–3840 | 960–1920 | 480–960 | 240–480 | 120–240 | 60–120 | 30–60 | 0–30 |

## 4. Electrical parameters (per order, per cycle)

| Parameter | Definition | Applies to |
| --- | --- | --- |
| module | $\lvert X_{o,c}\rvert$ | all |
| energy | $\lvert X_{o,c}\rvert^2$ | all |
| angle | $\arg X_{o,c}$ | DFT, Stockwell |
| THD | $\sqrt{\sum_{h=2}^{20} \lvert X_{h,c}\rvert^2} \,/\, \lvert X_{1,c}\rvert$ (one value per cycle, order `agg`) | DFT, Stockwell |
| impedance magnitude | $\lvert V_{o,c}\rvert / \lvert I_{o,c}\rvert$ | V/I pairs |
| V-I angle | $\arg(V_{o,c} I_{o,c}^*)$ | V/I pairs, DFT, Stockwell |

V/I pairs: fault voltage/fault current, negative-sequence V/I, voltage sum/neutral current.
Denominators are guarded with $\max(\cdot, 10^{-12})$.

## 5. Statistics (over the 10 cycles)

Within-window statistics use the **post-fault** window. Cross-window statistics compare the
post-fault mean with the pre-fault mean.

| Statistic | Definition |
| --- | --- |
| mean | $\bar p = \frac{1}{10}\sum_c p_c$ |
| max | $\max_c p_c$ |
| amplitude | $\max_c p_c - \min_c p_c$ |
| std | population standard deviation (ddof = 0) |
| kurtosis | unbiased Fisher excess kurtosis; 0 when the variance is numerically zero |
| entropy | $-\sum_c q_c \ln q_c$, with $q_c = \lvert p_c\rvert / \sum \lvert p\rvert$ |
| delta | $\bar p_\text{post} - \bar p_\text{pre}$ |
| ratio | $\bar p_\text{post} / \bar p_\text{pre}$ |

Angles are circular. Only mean ($\arg \overline{e^{j\theta}}$), std ($\sqrt{-2\ln R}$, where
$R = \lvert\overline{e^{j\theta}}\rvert$) and delta (the difference of circular means, wrapped to
$(-\pi, \pi]$) are defined for them.

A feature id reads `transform.channel.order.parameter.statistic`. For example,
`stockwell.neutral_current.h3.module.ratio` is the mean 3rd-harmonic Stockwell amplitude of the
neutral current after the fault divided by the same mean before it.

## 6. The selected features

The set used on the device (`thesis_train`) comes from the thesis's own selection chain, re-run on
the **training soils only** (`python/select_train.py`, 100 stability runs, φ = 0.256). π is the
fraction of runs that selected the feature's cluster. Unless stated otherwise, "after" means the
mean over the 10 post-fault cycles and "before" the mean over the 10 pre-fault cycles.

| # | π | Feature id | In words |
| ---: | ---: | --- | --- |
| 1 | 0.95 | `dft.fault_voltage__fault_current.h1.impedance_magnitude.max` | largest per-cycle fundamental impedance $\lvert V_1\rvert/\lvert I_1\rvert$ of the faulted phase after the fault |
| 2 | 0.76 | `wavelet.fault_voltage.d3.module.delta` | change (after − before) in the RMS of the faulted-phase voltage's 960–1920 Hz band |
| 3 | 0.68 | `stockwell.fault_current.agg.thd.ratio` | faulted-phase current THD (Stockwell, harmonics 2–20 over 1), after ÷ before |
| 4 | 0.48 | `wavelet.neutral_current.d7.module.max` | largest per-cycle RMS of the neutral current's 60–120 Hz band after the fault |
| 5 | 0.43 | `wavelet.fault_voltage__fault_current.d6.impedance_magnitude.ratio` | faulted-phase 120–240 Hz band impedance (RMS V ÷ RMS I), after ÷ before |
| 6 | 0.38 | `wavelet.fault_current.d8.energy.amplitude` | max − min over the post-fault cycles of the faulted-phase current's 30–60 Hz band energy (RMS²) |
| 7 | 0.35 | `dft.voltage_sum__neutral_current.h3.impedance_magnitude.mean` | mean 3rd-harmonic "zero-sequence impedance" $\lvert 3V_{0,3}\rvert/\lvert I_{N,3}\rvert$ after the fault |
| 8 | 0.29 | `stockwell.voltage_sum__neutral_current.h3.impedance_magnitude.ratio` | the same 3rd-harmonic impedance (Stockwell), after ÷ before |
| 9 | 0.25 | `dft.fault_voltage.h1.energy.max` | largest per-cycle squared fundamental amplitude of the faulted-phase voltage after the fault |
| 10 | 0.21 | `stockwell.fault_voltage.h8.energy.mean` | mean squared 8th-harmonic amplitude of the faulted-phase voltage after the fault |
| 11 | 0.21 | `stockwell.negative_sequence_voltage.h12.module.delta` | change in the 12th-harmonic negative-sequence voltage amplitude |
| 12 | 0.20 | `stockwell.neutral_current.h1.angle.mean` | circular mean of the neutral current's fundamental phase angle after the fault |
| 13 | 0.19 | `dft.fault_voltage.h1.angle.mean` | circular mean of the faulted-phase voltage's fundamental phase angle after the fault |
| 14 | 0.19 | `stockwell.negative_sequence_voltage.h10.module.delta` | change in the 10th-harmonic negative-sequence voltage amplitude |
| 15 | 0.15 | `dft.negative_sequence_voltage.h14.energy.mean` | mean squared 14th-harmonic negative-sequence voltage amplitude after the fault |

Angles are absolute phases relative to the window start. They carry information only because
every window starts at a fixed time relative to the fault instant.

**What the firmware has to compute** for this set, for both windows:

| Transform | Raw signals | Orders |
| --- | --- | --- |
| DFT (per-cycle 256-point) | $v_\varphi, i_\varphi$, $v_A, v_B, v_C$, $i_N$ | h1, h3, h14 |
| Stockwell (2560-point) | $i_\varphi$ (all h1–h20, for THD), $v_\varphi$, $v_A, v_B, v_C$, $i_N$ | h1, h3, h8, h10, h12, THD |
| Wavelet (SWT, 8 levels) | $v_\varphi$, $i_\varphi$, $i_N$ | d3, d6, d7, d8 |

None of the 15 needs the negative-sequence current, and no wavelet feature needs a negative
sequence, so the firmware needs no Hilbert transform for this set.

**Thesis's published ranking** (`thesis_published`): it is taken from Run B over all 7990
simulations, so its selection saw the test soils. It shares 10 of the 15 clusters above. The five
it has instead are `stockwell.neutral_current.h3.module.ratio`,
`wavelet.voltage_sum__neutral_current.d3.impedance_magnitude.max`,
`stockwell.fault_current.h14.energy.ratio`, `dft.neutral_current.h3.module.std` and
`stockwell.fault_current.h1.angle.mean`.

## 7. Float32 input sensitivity

The device receives float32 windows. Converting the test windows from float64 to float32 changes
the 15 features above by at most 9e-4 of their spread across files. The worst cases are the two
voltage-sum/neutral impedances: the voltage sum $v_A+v_B+v_C$ cancels three ~21 kV phases down to
a small residual. Over all 7056 features the median relative change is 6.7e-7. The largest effects
are again on voltage-sum/neutral impedances at high harmonics, up to 7% of the feature's spread.
