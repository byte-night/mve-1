/* ============================================================================
 * MVE-1: Monologue Voice Engine - DSP Implementation
 * 
 * First-party digital signal processing engine.
 * No external dependencies beyond libc/libm.
 * ============================================================================
 */

#include "mve_dsp.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ============================================================================
 * Utility Functions
 * ============================================================================
 */

int mve_is_power_of_2(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

int mve_next_power_of_2(int n) {
    if (n <= 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

int mve_log2_int(int n) {
    int log = 0;
    while ((1 << log) < n) log++;
    return (1 << log) == n ? log : -1;
}

float mve_hz_to_mel(float hz) {
    return 1127.0f * logf(1.0f + hz / 700.0f);
}

float mve_mel_to_hz(float mel) {
    return 700.0f * (expf(mel / 1127.0f) - 1.0f);
}

int mve_freq_to_bin(float freq, int sample_rate, int fft_size) {
    return (int)(freq / (sample_rate / 2) * (fft_size / 2));
}

float mve_bin_to_freq(int bin, int sample_rate, int fft_size) {
    return (float)bin / (fft_size / 2) * (sample_rate / 2);
}

/* ============================================================================
 * Window Functions
 * ============================================================================
 */

int mve_window_generate(float* window, int size, MVE_WindowType type, double param) {
    if (!window || size <= 0) return -1;
    
    (void)param;  /* Used for Kaiser, Blackman-Harris, etc. */
    
    for (int i = 0; i < size; i++) {
        switch (type) {
            case MVE_WINDOW_RECTANGULAR:
                window[i] = 1.0f;
                break;
            case MVE_WINDOW_HANN:
                window[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (size - 1)));
                break;
            case MVE_WINDOW_HAMMING:
                window[i] = 0.54f - 0.46f * cosf(2.0f * M_PI * i / (size - 1));
                break;
            case MVE_WINDOW_BLACKMAN:
                window[i] = 0.42f - 0.5f * cosf(2.0f * M_PI * i / (size - 1))
                                 + 0.08f * cosf(4.0f * M_PI * i / (size - 1));
                break;
            case MVE_WINDOW_BLACKMAN_HARRIS:
                window[i] = 0.35875f - 0.48829f * cosf(2.0f * M_PI * i / (size - 1))
                                   + 0.14128f * cosf(4.0f * M_PI * i / (size - 1))
                                   - 0.01168f * cosf(6.0f * M_PI * i / (size - 1));
                break;
            case MVE_WINDOW_NUTTALL:
                window[i] = 0.355768f - 0.487396f * cosf(2.0f * M_PI * i / (size - 1))
                                    + 0.144232f * cosf(4.0f * M_PI * i / (size - 1))
                                    - 0.012604f * cosf(6.0f * M_PI * i / (size - 1));
                break;
            case MVE_WINDOW_KAISER:
            default:
                /* Simplified Kaiser with fixed beta */
                {
                    float beta = 5.0f;
                    float alpha = (size - 1) / 2.0f;
                    float x = i - alpha;
                    float bessel = 1.0f;  /* Approximation */
                    float denom = 1.0f;
                    for (int k = 1; k < 20; k++) {
                        float term = (beta * beta * x * x) / (4.0f * k * k * alpha * alpha);
                        if (term < 1e-6f) break;
                        bessel += term;
                        if (k == 1) denom = bessel;
                    }
                    window[i] = bessel / denom;
                }
                break;
        }
    }
    return 0;
}

int mve_window_apply(float* buffer, const float* window, int size) {
    if (!buffer || !window || size <= 0) return -1;
    for (int i = 0; i < size; i++) {
        buffer[i] *= window[i];
    }
    return 0;
}

/* ============================================================================
 * Internal Helper Functions for FFT
 * ============================================================================ */

static void build_bit_reverse_table(int* table, int size, int log2_size) {
    for (int i = 0; i < size; i++) {
        int rev = 0;
        int tmp = i;
        for (int j = 0; j < log2_size; j++) {
            rev = (rev << 1) | (tmp & 1);
            tmp >>= 1;
        }
        table[i] = rev;
    }
}

static void build_twiddle_tables(float* cos_tab, float* sin_tab, int size) {
    for (int i = 0; i < size / 2; i++) {
        double angle = -2.0 * M_PI * i / size;
        cos_tab[i] = (float)cos(angle);
        sin_tab[i] = (float)sin(angle);
    }
}

/* Check if n is prime */
static int is_prime(int n) {
    if (n < 2) return 0;
    if (n == 2) return 1;
    if (n % 2 == 0) return 0;
    for (int i = 3; i * i <= n; i += 2) {
        if (n % i == 0) return 0;
    }
    return 1;
}

/* Bluestein's algorithm for arbitrary-size FFT */
static int bluestein_fft_forward(MVE_FFTPlan* plan, float* re, float* im) {
    int n = plan->n;
    
    /* Find next power of 2 >= 2*n - 1 */
    int m = 1;
    while (m < 2 * n - 1) m *= 2;
    
    /* Create chirp sequence: exp(-j*pi*k^2/n) for k = 0..n-1 */
    float* chirp_re = (float*)calloc(m, sizeof(float));
    float* chirp_im = (float*)calloc(m, sizeof(float));
    if (!chirp_re || !chirp_im) {
        free(chirp_re);
        free(chirp_im);
        return -1;
    }
    
    for (int k = 0; k < n; k++) {
        double angle = -M_PI * (double)(k * k) / n;
        chirp_re[k] = (float)cos(angle);
        chirp_im[k] = (float)sinf(angle);
    }
    
    /* Create extended input: x[k] * chirp[k] for k = 0..n-1, rest zero */
    float* ext_re = (float*)calloc(m, sizeof(float));
    float* ext_im = (float*)calloc(m, sizeof(float));
    if (!ext_re || !ext_im) {
        free(chirp_re);
        free(chirp_im);
        free(ext_re);
        free(ext_im);
        return -1;
    }
    
    for (int k = 0; k < n; k++) {
        /* Complex multiply: input[k] * chirp[k] */
        ext_re[k] = re[k] * chirp_re[k] - im[k] * chirp_im[k];
        ext_im[k] = re[k] * chirp_im[k] + im[k] * chirp_re[k];
    }
    
    /* Create time-reversed conjugate chirp for convolution */
    float* chirp_rev_re = (float*)calloc(m, sizeof(float));
    float* chirp_rev_im = (float*)calloc(m, sizeof(float));
    if (!chirp_rev_re || !chirp_rev_im) {
        free(chirp_re);
        free(chirp_im);
        free(ext_re);
        free(ext_im);
        free(chirp_rev_re);
        free(chirp_rev_im);
        return -1;
    }
    
    chirp_rev_re[0] = chirp_re[0];
    chirp_rev_im[0] = chirp_im[0];
    for (int k = 1; k < n; k++) {
        chirp_rev_re[k] = chirp_re[k];
        chirp_rev_im[k] = -chirp_im[k];  /* Conjugate */
        chirp_rev_re[m - k] = chirp_re[k];
        chirp_rev_im[m - k] = -chirp_im[k];
    }
    
    /* FFT of extended input and chirp */
    MVE_FFTPlan* fft_m = mve_fft_create(m);
    if (!fft_m) {
        free(chirp_re);
        free(chirp_im);
        free(ext_re);
        free(ext_im);
        free(chirp_rev_re);
        free(chirp_rev_im);
        return -1;
    }
    
    mve_fft_forward(fft_m, ext_re, ext_im);
    mve_fft_forward(fft_m, chirp_rev_re, chirp_rev_im);
    
    /* Complex multiply in frequency domain */
    float* prod_re = (float*)malloc(m * sizeof(float));
    float* prod_im = (float*)malloc(m * sizeof(float));
    if (!prod_re || !prod_im) {
        mve_fft_destroy(fft_m);
        free(chirp_re);
        free(chirp_im);
        free(ext_re);
        free(ext_im);
        free(chirp_rev_re);
        free(chirp_rev_im);
        return -1;
    }
    
    for (int i = 0; i < m; i++) {
        prod_re[i] = ext_re[i] * chirp_rev_re[i] - ext_im[i] * chirp_rev_im[i];
        prod_im[i] = ext_re[i] * chirp_rev_im[i] + ext_im[i] * chirp_rev_re[i];
    }
    
    /* Inverse FFT */
    mve_fft_inverse(fft_m, prod_re, prod_im);
    mve_fft_destroy(fft_m);
    
    /* Multiply by chirp again and extract result */
    for (int k = 0; k < n; k++) {
        float tmp_re = prod_re[k] * chirp_re[k] - prod_im[k] * chirp_im[k];
        float tmp_im = prod_re[k] * chirp_im[k] + prod_im[k] * chirp_re[k];
        re[k] = tmp_re;
        im[k] = tmp_im;
    }
    
    /* Cleanup */
    free(chirp_re);
    free(chirp_im);
    free(ext_re);
    free(ext_im);
    free(chirp_rev_re);
    free(chirp_rev_im);
    free(prod_re);
    free(prod_im);
    
    return 0;
}

/* Bluestein's algorithm for inverse FFT */
static int bluestein_fft_inverse(MVE_FFTPlan* plan, float* re, float* im) {
    int n = plan->n;
    
    /* Conjugate input */
    for (int i = 0; i < n; i++) {
        im[i] = -im[i];
    }
    
    /* Forward FFT using Bluestein */
    int result = bluestein_fft_forward(plan, re, im);
    if (result != 0) return result;
    
    /* Conjugate again and scale */
    float scale = 1.0f / n;
    for (int i = 0; i < n; i++) {
        re[i] = re[i] * scale;
        im[i] = -im[i] * scale;
    }
    
    return 0;
}

/* ============================================================================
 * FFT Engine Implementation
 * ============================================================================ */

MVE_FFTPlan* mve_fft_create(int n) {
    if (n < MVE_DSP_MIN_FFT_SIZE || n > MVE_DSP_MAX_FFT_SIZE) return NULL;
    
    MVE_FFTPlan* plan = (MVE_FFTPlan*)calloc(1, sizeof(MVE_FFTPlan));
    if (!plan) return NULL;
    
    plan->n = n;
    plan->is_power_of_2 = mve_is_power_of_2(n);
    plan->owns_tables = 1;
    
    if (plan->is_power_of_2) {
        /* Power-of-2: use radix-2 Cooley-Tukey */
        plan->log2n = mve_log2_int(n);
        if (plan->log2n < 0) {
            free(plan);
            return NULL;
        }
        
        plan->bit_reverse_table = (int*)malloc(n * sizeof(int));
        plan->cos_table = (float*)malloc((n / 2) * sizeof(float));
        plan->sin_table = (float*)malloc((n / 2) * sizeof(float));
        
        if (!plan->bit_reverse_table || !plan->cos_table || !plan->sin_table) {
            mve_fft_destroy(plan);
            return NULL;
        }
        
        build_bit_reverse_table(plan->bit_reverse_table, n, plan->log2n);
        build_twiddle_tables(plan->cos_table, plan->sin_table, n);
        plan->bluestein_fft_plan = NULL;
    } else {
        /* Non-power-of-2: use Bluestein's algorithm */
        plan->log2n = 0;
        plan->bit_reverse_table = NULL;
        plan->cos_table = NULL;
        plan->sin_table = NULL;
        
        /* Create nested FFT plan for Bluestein (next power of 2) */
        int m = 1;
        while (m < 2 * n - 1) m *= 2;
        plan->bluestein_fft_plan = mve_fft_create(m);
        
        if (!plan->bluestein_fft_plan) {
            mve_fft_destroy(plan);
            return NULL;
        }
    }
    
    return plan;
}

void mve_fft_destroy(MVE_FFTPlan* plan) {
    if (!plan) return;
    if (plan->owns_tables) {
        free(plan->bit_reverse_table);
        free(plan->cos_table);
        free(plan->sin_table);
    }
    /* Destroy nested Bluestein FFT plan if present */
    if (plan->bluestein_fft_plan) {
        mve_fft_destroy(plan->bluestein_fft_plan);
    }
    free(plan);
}

static void butterfly_radix2(float* re, float* im, int n,
                             const int* bit_rev, const float* cos_tab,
                             const float* sin_tab) {
    /* Bit-reversal permutation */
    for (int i = 0; i < n; i++) {
        int j = bit_rev[i];
        if (i < j) {
            float tmp_r = re[i]; re[i] = re[j]; re[j] = tmp_r;
            float tmp_i = im[i]; im[i] = im[j]; im[j] = tmp_i;
        }
    }
    
    /* Cooley-Tukey radix-2 FFT */
    int m = 1;
    for (int stage = 0; stage < mve_log2_int(n); stage++) {
        int half_m = m;
        m *= 2;
        
        for (int k = 0; k < n; k += m) {
            for (int j = 0; j < half_m; j++) {
                int idx1 = k + j;
                int idx2 = k + j + half_m;
                int twiddle_idx = j * (n / m);
                
                float cos_val = cos_tab[twiddle_idx];
                float sin_val = sin_tab[twiddle_idx];
                
                float tr = cos_val * re[idx2] - sin_val * im[idx2];
                float ti = cos_val * im[idx2] + sin_val * re[idx2];
                
                re[idx2] = re[idx1] - tr;
                im[idx2] = im[idx1] - ti;
                re[idx1] = re[idx1] + tr;
                im[idx1] = im[idx1] + ti;
            }
        }
    }
}

int mve_fft_forward(MVE_FFTPlan* plan, float* re, float* im) {
    if (!plan || !re || !im) return -1;
    
    if (plan->is_power_of_2) {
        /* Power-of-2: use radix-2 Cooley-Tukey */
        butterfly_radix2(re, im, plan->n, plan->bit_reverse_table, 
                         plan->cos_table, plan->sin_table);
    } else {
        /* Non-power-of-2: use Bluestein's algorithm */
        return bluestein_fft_forward(plan, re, im);
    }
    return 0;
}

int mve_fft_inverse(MVE_FFTPlan* plan, float* re, float* im) {
    if (!plan || !re || !im) return -1;
    
    if (plan->is_power_of_2) {
        /* Power-of-2: use radix-2 Cooley-Tukey */
        /* Conjugate input */
        for (int i = 0; i < plan->n; i++) {
            im[i] = -im[i];
        }
        
        butterfly_radix2(re, im, plan->n, plan->bit_reverse_table,
                         plan->cos_table, plan->sin_table);
        
        /* Conjugate again and scale */
        float scale = 1.0f / plan->n;
        for (int i = 0; i < plan->n; i++) {
            re[i] = re[i] * scale;
            im[i] = -im[i] * scale;
        }
    } else {
        /* Non-power-of-2: use Bluestein's algorithm */
        return bluestein_fft_inverse(plan, re, im);
    }
    
    return 0;
}

int mve_rfft_forward(MVE_FFTPlan* plan, float* input, float* output) {
    if (!plan || !input || !output) return -1;
    
    /* Copy to real/imag buffers */
    float* re = (float*)malloc(plan->n * sizeof(float));
    float* im = (float*)calloc(plan->n, sizeof(float));
    if (!re || !im) { free(re); free(im); return -1; }
    
    memcpy(re, input, plan->n * sizeof(float));
    
    mve_fft_forward(plan, re, im);
    
    /* Pack output: [DC_re, DC_im, bin1_re, bin1_im, ..., Nyquist_re, Nyquist_im] */
    output[0] = re[0];
    output[1] = im[0];
    for (int i = 1; i < plan->n / 2; i++) {
        output[2*i] = re[i];
        output[2*i+1] = im[i];
    }
    output[plan->n] = re[plan->n/2];
    output[plan->n+1] = im[plan->n/2];
    
    free(re);
    free(im);
    return 0;
}

int mve_rfft_inverse(MVE_FFTPlan* plan, float* input, float* output) {
    if (!plan || !input || !output) return -1;
    
    float* re = (float*)calloc(plan->n, sizeof(float));
    float* im = (float*)calloc(plan->n, sizeof(float));
    if (!re || !im) { free(re); free(im); return -1; }
    
    /* Unpack input */
    re[0] = input[0];
    im[0] = input[1];
    for (int i = 1; i < plan->n / 2; i++) {
        re[i] = input[2*i];
        im[i] = input[2*i+1];
    }
    re[plan->n/2] = input[plan->n];
    im[plan->n/2] = input[plan->n+1];
    
    mve_fft_inverse(plan, re, im);
    
    memcpy(output, re, plan->n * sizeof(float));
    
    free(re);
    free(im);
    return 0;
}

int mve_power_spectrum(const float* re, const float* im, int n, float* power) {
    if (!re || !im || !power || n <= 0) return -1;
    for (int i = 0; i < n; i++) {
        power[i] = re[i] * re[i] + im[i] * im[i];
    }
    return 0;
}

int mve_magnitude_spectrum(const float* re, const float* im, int n, float* mag) {
    if (!re || !im || !mag || n <= 0) return -1;
    for (int i = 0; i < n; i++) {
        mag[i] = sqrtf(re[i] * re[i] + im[i] * im[i]);
    }
    return 0;
}

int mve_log_magnitude_spectrum(const float* re, const float* im, int n, 
                                float* log_mag, float eps) {
    if (!re || !im || !log_mag || n <= 0) return -1;
    for (int i = 0; i < n; i++) {
        float mag = sqrtf(re[i] * re[i] + im[i] * im[i]);
        log_mag[i] = logf(fmaxf(mag, eps));
    }
    return 0;
}

/* ============================================================================
 * STFT Engine Implementation
 * ============================================================================
 */

MVE_STFTConfig* mve_stft_config_create(int fft_size, int hop_size, int win_length,
                                        MVE_WindowType window_type) {
    if (!mve_is_power_of_2(fft_size)) return NULL;
    if (hop_size <= 0 || hop_size > fft_size) return NULL;
    if (win_length <= 0 || win_length > fft_size) return NULL;
    
    MVE_STFTConfig* config = (MVE_STFTConfig*)calloc(1, sizeof(MVE_STFTConfig));
    if (!config) return NULL;
    
    config->fft_plan = mve_fft_create(fft_size);
    if (!config->fft_plan) {
        free(config);
        return NULL;
    }
    
    config->window = (float*)malloc(win_length * sizeof(float));
    if (!config->window) {
        mve_stft_config_destroy(config);
        return NULL;
    }
    
    mve_window_generate(config->window, win_length, window_type, 0.0);
    
    config->fft_size = fft_size;
    config->hop_size = hop_size;
    config->win_length = win_length;
    config->window_type = window_type;
    config->num_freq_bins = fft_size / 2 + 1;
    
    config->frame_buffer = (float*)calloc(fft_size, sizeof(float));
    config->overlap_buffer = (float*)calloc(fft_size, sizeof(float));
    config->window_sum = (float*)calloc(fft_size, sizeof(float));
    
    if (!config->frame_buffer || !config->overlap_buffer || !config->window_sum) {
        mve_stft_config_destroy(config);
        return NULL;
    }
    
    return config;
}

void mve_stft_config_destroy(MVE_STFTConfig* config) {
    if (!config) return;
    mve_fft_destroy(config->fft_plan);
    free(config->window);
    free(config->frame_buffer);
    free(config->overlap_buffer);
    free(config->window_sum);
    free(config);
}

int mve_stft_forward(MVE_STFTConfig* config, const float* input, int input_samples,
                     float* spectrogram_re, float* spectrogram_im) {
    if (!config || !input || !spectrogram_re || !spectrogram_im || input_samples <= 0) {
        return -1;
    }
    
    config->num_frames = 1 + (input_samples - config->win_length) / config->hop_size;
    int num_bins = config->num_freq_bins;
    
    for (int frame = 0; frame < config->num_frames; frame++) {
        int sample_offset = frame * config->hop_size;
        
        /* Copy frame and apply window */
        memset(config->frame_buffer, 0, config->fft_size * sizeof(float));
        for (int i = 0; i < config->win_length && (sample_offset + i) < input_samples; i++) {
            config->frame_buffer[i] = input[sample_offset + i] * config->window[i];
        }
        
        /* Zero-pad imaginary part */
        float* im = (float*)calloc(config->fft_size, sizeof(float));
        if (!im) return -1;
        
        mve_fft_forward(config->fft_plan, config->frame_buffer, im);
        
        /* Store only positive frequencies */
        for (int bin = 0; bin < num_bins; bin++) {
            spectrogram_re[frame * num_bins + bin] = config->frame_buffer[bin];
            spectrogram_im[frame * num_bins + bin] = im[bin];
        }
        
        free(im);
    }
    
    return 0;
}

int mve_stft_inverse(MVE_STFTConfig* config, const float* spectrogram_re,
                     const float* spectrogram_im, int num_frames,
                     float* output, int output_samples) {
    if (!config || !spectrogram_re || !spectrogram_im || !output || 
        num_frames <= 0 || output_samples <= 0) {
        return -1;
    }
    
    int num_bins = config->num_freq_bins;
    int synth_samples = (num_frames - 1) * config->hop_size + config->win_length;
    
    if (synth_samples > output_samples) {
        return -1;
    }
    
    memset(output, 0, synth_samples * sizeof(float));
    memset(config->window_sum, 0, synth_samples * sizeof(float));
    
    for (int frame = 0; frame < num_frames; frame++) {
        int sample_offset = frame * config->hop_size;
        
        /* Reconstruct full spectrum */
        float* re = (float*)calloc(config->fft_size, sizeof(float));
        float* im = (float*)calloc(config->fft_size, sizeof(float));
        if (!re || !im) { free(re); free(im); return -1; }
        
        /* Copy positive frequencies */
        for (int bin = 0; bin < num_bins; bin++) {
            re[bin] = spectrogram_re[frame * num_bins + bin];
            im[bin] = spectrogram_im[frame * num_bins + bin];
        }
        
        /* Fill negative frequencies (Hermitian symmetry) */
        for (int bin = num_bins; bin < config->fft_size; bin++) {
            int sym_bin = config->fft_size - bin;
            re[bin] = re[sym_bin];
            im[bin] = -im[sym_bin];
        }
        
        /* IFFT */
        mve_fft_inverse(config->fft_plan, re, im);
        
        /* Apply window and overlap-add */
        for (int i = 0; i < config->win_length && (sample_offset + i) < synth_samples; i++) {
            output[sample_offset + i] += re[i] * config->window[i];
            config->window_sum[sample_offset + i] += config->window[i] * config->window[i];
        }
        
        free(re);
        free(im);
    }
    
    /* Normalize by window sum */
    for (int i = 0; i < synth_samples; i++) {
        if (config->window_sum[i] > 1e-6f) {
            output[i] /= config->window_sum[i];
        }
    }
    
    return 0;
}

int mve_stft_backward(MVE_STFTConfig* config, const float* grad_spectrogram_re,
                      const float* grad_spectrogram_im, int num_frames,
                      float* grad_input) {
    /* Placeholder for backprop gradient computation */
    (void)config;
    (void)grad_spectrogram_re;
    (void)grad_spectrogram_im;
    (void)num_frames;
    (void)grad_input;
    return 0;
}

/* ============================================================================
 * Mel Filterbank Implementation
 * ============================================================================
 */

MVE_MelFilterbank* mve_mel_filterbank_create(int sample_rate, int n_fft, int n_mels,
                                              float fmin, float fmax) {
    if (sample_rate <= 0 || n_fft <= 0 || n_mels <= 0) return NULL;
    if (fmin < 0 || fmax > sample_rate / 2 || fmin >= fmax) return NULL;
    
    MVE_MelFilterbank* fb = (MVE_MelFilterbank*)calloc(1, sizeof(MVE_MelFilterbank));
    if (!fb) return NULL;
    
    int num_bins = n_fft / 2 + 1;
    
    fb->filterbank = (float*)calloc(n_mels * num_bins, sizeof(float));
    fb->filter_starts = (int*)malloc(n_mels * sizeof(int));
    fb->filter_ends = (int*)malloc(n_mels * sizeof(int));
    
    if (!fb->filterbank || !fb->filter_starts || !fb->filter_ends) {
        mve_mel_filterbank_destroy(fb);
        return NULL;
    }
    
    fb->sample_rate = sample_rate;
    fb->n_fft = n_fft;
    fb->n_mels = n_mels;
    fb->fmin = fmin;
    fb->fmax = fmax;
    
    /* Compute mel filter centers */
    float mel_min = mve_hz_to_mel(fmin);
    float mel_max = mve_hz_to_mel(fmax);
    float mel_step = (mel_max - mel_min) / (n_mels + 1);
    
    for (int i = 0; i < n_mels; i++) {
        float center_mel = mel_min + (i + 1) * mel_step;
        float left_mel = center_mel - mel_step;
        float right_mel = center_mel + mel_step;
        
        float left_hz = mve_mel_to_hz(left_mel);
        float center_hz = mve_mel_to_hz(center_mel);
        float right_hz = mve_mel_to_hz(right_mel);
        
        int left_bin = mve_freq_to_bin(left_hz, sample_rate, n_fft);
        int center_bin = mve_freq_to_bin(center_hz, sample_rate, n_fft);
        int right_bin = mve_freq_to_bin(right_hz, sample_rate, n_fft);
        
        left_bin = (left_bin < 0) ? 0 : (left_bin >= num_bins ? num_bins - 1 : left_bin);
        center_bin = (center_bin < 0) ? 0 : (center_bin >= num_bins ? num_bins - 1 : center_bin);
        right_bin = (right_bin < 0) ? 0 : (right_bin >= num_bins ? num_bins - 1 : right_bin);
        
        fb->filter_starts[i] = left_bin;
        fb->filter_ends[i] = right_bin;
        
        /* Build triangular filter */
        for (int bin = left_bin; bin < center_bin; bin++) {
            float bin_hz = mve_bin_to_freq(bin, sample_rate, n_fft);
            if (bin_hz >= left_hz && bin_hz <= center_hz) {
                fb->filterbank[i * num_bins + bin] = (bin_hz - left_hz) / (center_hz - left_hz + 1e-6f);
            }
        }
        for (int bin = center_bin; bin <= right_bin; bin++) {
            float bin_hz = mve_bin_to_freq(bin, sample_rate, n_fft);
            if (bin_hz >= center_hz && bin_hz <= right_hz) {
                fb->filterbank[i * num_bins + bin] = (right_hz - bin_hz) / (right_hz - center_hz + 1e-6f);
            }
        }
    }
    
    return fb;
}

void mve_mel_filterbank_destroy(MVE_MelFilterbank* fb) {
    if (!fb) return;
    free(fb->filterbank);
    free(fb->filter_starts);
    free(fb->filter_ends);
    free(fb);
}

int mve_spectrogram_to_mel(const MVE_MelFilterbank* fb, const float* spectrogram,
                           int n_frames, float* mel_spec) {
    if (!fb || !spectrogram || !mel_spec || n_frames <= 0) return -1;
    
    int num_bins = fb->n_fft / 2 + 1;
    
    for (int frame = 0; frame < n_frames; frame++) {
        for (int i = 0; i < fb->n_mels; i++) {
            float energy = 0.0f;
            for (int bin = fb->filter_starts[i]; bin <= fb->filter_ends[i]; bin++) {
                energy += spectrogram[frame * num_bins + bin] * fb->filterbank[i * num_bins + bin];
            }
            mel_spec[frame * fb->n_mels + i] = fmaxf(energy, 1e-10f);
        }
    }
    
    return 0;
}

int mve_mel_to_logmel(float* mel_spec, int n_frames, int n_mels, float eps) {
    if (!mel_spec || n_frames <= 0 || n_mels <= 0) return -1;
    
    for (int i = 0; i < n_frames * n_mels; i++) {
        mel_spec[i] = logf(fmaxf(mel_spec[i], eps));
    }
    
    return 0;
}

/* ============================================================================
 * F0 Extraction (YIN Algorithm Variant)
 * ============================================================================
 */

MVE_F0Extractor* mve_f0_extractor_create(int sample_rate, int frame_size, int hop_size,
                                          float f0_min, float f0_max) {
    if (sample_rate <= 0 || frame_size <= 0 || hop_size <= 0) return NULL;
    if (f0_min <= 0 || f0_max <= f0_min) return NULL;
    
    MVE_F0Extractor* ext = (MVE_F0Extractor*)calloc(1, sizeof(MVE_F0Extractor));
    if (!ext) return NULL;
    
    ext->diff_buffer = (float*)malloc(frame_size * sizeof(float));
    ext->cumsum_buffer = (float*)malloc(frame_size * sizeof(float));
    ext->parabolic_buffer = (float*)malloc(frame_size * sizeof(float));
    
    if (!ext->diff_buffer || !ext->cumsum_buffer || !ext->parabolic_buffer) {
        mve_f0_extractor_destroy(ext);
        return NULL;
    }
    
    ext->sample_rate = sample_rate;
    ext->frame_size = frame_size;
    ext->hop_size = hop_size;
    ext->f0_min = f0_min;
    ext->f0_max = f0_max;
    ext->voicing_threshold = 0.1f;
    
    return ext;
}

void mve_f0_extractor_destroy(MVE_F0Extractor* ext) {
    if (!ext) return;
    free(ext->diff_buffer);
    free(ext->cumsum_buffer);
    free(ext->parabolic_buffer);
    free(ext);
}

int mve_f0_extract(MVE_F0Extractor* ext, const float* audio, int n_samples,
                   float* f0_contour, float* voicing_prob, int* n_frames) {
    if (!ext || !audio || !f0_contour || !voicing_prob || !n_frames || n_samples <= 0) {
        return -1;
    }
    
    int min_period = (int)(ext->sample_rate / ext->f0_max);
    int max_period = (int)(ext->sample_rate / ext->f0_min);
    
    if (min_period < 2) min_period = 2;
    if (max_period > ext->frame_size / 4) max_period = ext->frame_size / 4;
    
    *n_frames = 1 + (n_samples - ext->frame_size) / ext->hop_size;
    
    for (int frame = 0; frame < *n_frames; frame++) {
        int start = frame * ext->hop_size;
        if (start + ext->frame_size > n_samples) {
            f0_contour[frame] = 0.0f;
            voicing_prob[frame] = 0.0f;
            continue;
        }
        
        /* Simplified autocorrelation-based F0 estimation */
        float best_corr = -1.0f;
        int best_period = 0;
        
        for (int period = min_period; period <= max_period; period++) {
            float corr = 0.0f;
            float energy = 0.0f;
            
            for (int i = 0; i < period && (start + i + period) < n_samples; i++) {
                corr += audio[start + i] * audio[start + i + period];
                energy += audio[start + i] * audio[start + i];
            }
            
            if (energy > 1e-6f) {
                corr /= energy;
            }
            
            if (corr > best_corr) {
                best_corr = corr;
                best_period = period;
            }
        }
        
        if (best_corr > ext->voicing_threshold && best_period > 0) {
            f0_contour[frame] = ext->sample_rate / best_period;
            voicing_prob[frame] = best_corr;
        } else {
            f0_contour[frame] = 0.0f;
            voicing_prob[frame] = 0.0f;
        }
    }
    
    return 0;
}

int mve_f0_extract_pyin(MVE_F0Extractor* ext, const float* audio, int n_samples,
                        float* f0_contour, float* voicing_prob, int* n_frames) {
    /* Placeholder for PYIN implementation */
    return mve_f0_extract(ext, audio, n_samples, f0_contour, voicing_prob, n_frames);
}

/* ============================================================================
 * Energy Extraction
 * ============================================================================
 */

int mve_energy_rms(const float* audio, int n_samples, int frame_size, int hop_size,
                   float* energy, int* n_frames) {
    if (!audio || !energy || !n_frames || n_samples <= 0 || frame_size <= 0 || hop_size <= 0) {
        return -1;
    }
    
    *n_frames = 1 + (n_samples - frame_size) / hop_size;
    
    for (int frame = 0; frame < *n_frames; frame++) {
        int start = frame * hop_size;
        float sum = 0.0f;
        
        for (int i = 0; i < frame_size && (start + i) < n_samples; i++) {
            sum += audio[start + i] * audio[start + i];
        }
        
        energy[frame] = sqrtf(sum / frame_size);
    }
    
    return 0;
}

int mve_energy_db(const float* audio, int n_samples, int frame_size, int hop_size,
                  float* energy_db, int* n_frames, float ref_level) {
    if (mve_energy_rms(audio, n_samples, frame_size, hop_size, energy_db, n_frames) != 0) {
        return -1;
    }
    
    for (int i = 0; i < *n_frames; i++) {
        energy_db[i] = 20.0f * log10f(fmaxf(energy_db[i], ref_level));
    }
    
    return 0;
}

/* ============================================================================
 * Resampling
 * ============================================================================
 */

MVE_Resampler* mve_resampler_create(int input_rate, int output_rate, int filter_order) {
    if (input_rate <= 0 || output_rate <= 0 || filter_order <= 0) return NULL;
    
    MVE_Resampler* res = (MVE_Resampler*)calloc(1, sizeof(MVE_Resampler));
    if (!res) return NULL;
    
    res->input_rate = input_rate;
    res->output_rate = output_rate;
    res->filter_order = filter_order;
    
    /* Simple ratio computation */
    res->lFactor = output_rate;
    res->mFactor = input_rate;
    
    /* Reduce fraction */
    int a = res->lFactor, b = res->mFactor;
    while (b != 0) { int t = b; b = a % b; a = t; }
    int gcd = a;
    res->lFactor /= gcd;
    res->mFactor /= gcd;
    
    res->state_buffer = (float*)malloc(filter_order * sizeof(float));
    if (!res->state_buffer) {
        mve_resampler_destroy(res);
        return NULL;
    }
    memset(res->state_buffer, 0, filter_order * sizeof(float));
    res->state_index = 0;
    
    return res;
}

void mve_resampler_destroy(MVE_Resampler* res) {
    if (!res) return;
    free(res->state_buffer);
    free(res->polyphase_filters);
    free(res);
}

int mve_resample(MVE_Resampler* res, const float* input, int n_input,
                 float* output, int* n_output) {
    if (!res || !input || !output || !n_output || n_input <= 0) return -1;
    
    if (res->input_rate == res->output_rate) {
        memcpy(output, input, n_input * sizeof(float));
        *n_output = n_input;
        return 0;
    }
    
    /* Simple linear interpolation resampler */
    float ratio = (float)res->output_rate / res->input_rate;
    int n_out = (int)(n_input * ratio);
    
    for (int i = 0; i < n_out; i++) {
        float src_pos = i / ratio;
        int src_idx = (int)src_pos;
        float frac = src_pos - src_idx;
        
        if (src_idx >= n_input - 1) {
            output[i] = input[n_input - 1];
        } else if (src_idx < 0) {
            output[i] = input[0];
        } else {
            output[i] = input[src_idx] * (1 - frac) + input[src_idx + 1] * frac;
        }
    }
    
    *n_output = n_out;
    return 0;
}

int mve_resample_stream(MVE_Resampler* res, const float* input, int n_input,
                        float* output, int max_output, int* n_output,
                        int* end_of_stream) {
    /* Placeholder for streaming resampler */
    (void)res; (void)input; (void)n_input;
    (void)output; (void)max_output; (void)n_output; (void)end_of_stream;
    return 0;
}

/* ============================================================================
 * Silence Detection
 * ============================================================================
 */

int mve_silence_detect(MVE_SilenceDetector* det, const float* audio, int n_samples,
                       int* silence_starts, int* silence_ends, int* n_regions,
                       int max_regions) {
    if (!det || !audio || !silence_starts || !silence_ends || !n_regions || n_samples <= 0) {
        return -1;
    }
    
    float threshold = powf(10.0f, det->threshold_db / 20.0f);
    int frame_size = det->frame_size;
    int hop_size = det->hop_size;
    int n_frames = 1 + (n_samples - frame_size) / hop_size;
    
    *n_regions = 0;
    int in_silence = 0;
    int silence_start = 0;
    int silence_frame_count = 0;
    
    for (int frame = 0; frame < n_frames; frame++) {
        int start = frame * hop_size;
        float rms = 0.0f;
        
        for (int i = 0; i < frame_size && (start + i) < n_samples; i++) {
            rms += audio[start + i] * audio[start + i];
        }
        rms = sqrtf(rms / frame_size);
        
        int is_silent = (rms < threshold);
        
        if (is_silent) {
            silence_frame_count++;
            if (!in_silence) {
                in_silence = 1;
                silence_start = start;
            }
        } else {
            if (in_silence && silence_frame_count >= det->min_silence_frames) {
                if (*n_regions < max_regions) {
                    silence_starts[*n_regions] = silence_start;
                    silence_ends[*n_regions] = start;
                    (*n_regions)++;
                }
            }
            in_silence = 0;
            silence_frame_count = 0;
        }
    }
    
    return 0;
}

int mve_silence_trim(MVE_SilenceDetector* det, const float* audio, int n_samples,
                     int* start_sample, int* end_sample) {
    if (!det || !audio || !start_sample || !end_sample || n_samples <= 0) {
        return -1;
    }
    
    float threshold = powf(10.0f, det->threshold_db / 20.0f);
    int frame_size = det->frame_size;
    int hop_size = det->hop_size;
    
    /* Find first non-silent frame */
    *start_sample = 0;
    for (int frame = 0; frame < n_samples / hop_size; frame++) {
        int start = frame * hop_size;
        float rms = 0.0f;
        
        for (int i = 0; i < frame_size && (start + i) < n_samples; i++) {
            rms += audio[start + i] * audio[start + i];
        }
        rms = sqrtf(rms / frame_size);
        
        if (rms >= threshold) {
            *start_sample = start;
            break;
        }
    }
    
    /* Find last non-silent frame */
    *end_sample = n_samples;
    for (int frame = n_samples / hop_size - 1; frame >= 0; frame--) {
        int start = frame * hop_size;
        float rms = 0.0f;
        
        for (int i = 0; i < frame_size && (start + i) < n_samples; i++) {
            rms += audio[start + i] * audio[start + i];
        }
        rms = sqrtf(rms / frame_size);
        
        if (rms >= threshold) {
            *end_sample = start + frame_size;
            break;
        }
    }
    
    return 0;
}

/* ============================================================================
 * Normalization
 * ============================================================================
 */

int mve_normalize_peak(float* audio, int n_samples, float target_peak) {
    if (!audio || n_samples <= 0 || target_peak <= 0 || target_peak > 1.0f) {
        return -1;
    }
    
    float max_peak = 0.0f;
    for (int i = 0; i < n_samples; i++) {
        float abs_val = fabsf(audio[i]);
        if (abs_val > max_peak) max_peak = abs_val;
    }
    
    if (max_peak > 1e-6f) {
        float scale = target_peak / max_peak;
        for (int i = 0; i < n_samples; i++) {
            audio[i] *= scale;
        }
    }
    
    return 0;
}

int mve_normalize_rms(float* audio, int n_samples, float target_rms) {
    if (!audio || n_samples <= 0 || target_rms <= 0) {
        return -1;
    }
    
    float current_rms = 0.0f;
    for (int i = 0; i < n_samples; i++) {
        current_rms += audio[i] * audio[i];
    }
    current_rms = sqrtf(current_rms / n_samples);
    
    if (current_rms > 1e-6f) {
        float scale = target_rms / current_rms;
        for (int i = 0; i < n_samples; i++) {
            audio[i] *= scale;
        }
    }
    
    return 0;
}

int mve_normalize_loudness(float* audio, int n_samples, int sample_rate, float target_lufs) {
    /* Simplified loudness normalization placeholder */
    (void)sample_rate;
    (void)target_lufs;
    return mve_normalize_rms(audio, n_samples, 0.1f);
}

/* ============================================================================
 * Spectral Features
 * ============================================================================
 */

int mve_spectral_centroid(const float* magnitudes, int n_bins, float* centroid) {
    if (!magnitudes || n_bins <= 0 || !centroid) return -1;
    
    float numerator = 0.0f;
    float denominator = 0.0f;
    
    for (int i = 0; i < n_bins; i++) {
        numerator += i * magnitudes[i];
        denominator += magnitudes[i];
    }
    
    *centroid = (denominator > 1e-6f) ? (numerator / denominator) : 0.0f;
    return 0;
}

int mve_spectral_bandwidth(const float* magnitudes, int n_bins, float centroid,
                           float* bandwidth) {
    if (!magnitudes || n_bins <= 0 || !bandwidth) return -1;
    
    float sum = 0.0f;
    float weight_sum = 0.0f;
    
    for (int i = 0; i < n_bins; i++) {
        float diff = i - centroid;
        sum += magnitudes[i] * diff * diff;
        weight_sum += magnitudes[i];
    }
    
    *bandwidth = (weight_sum > 1e-6f) ? sqrtf(sum / weight_sum) : 0.0f;
    return 0;
}

int mve_spectral_rolloff(const float* magnitudes, int n_bins, float rolloff_percent,
                         int* rolloff_bin) {
    if (!magnitudes || n_bins <= 0 || !rolloff_bin || 
        rolloff_percent <= 0 || rolloff_percent >= 1) {
        return -1;
    }
    
    float total_energy = 0.0f;
    for (int i = 0; i < n_bins; i++) {
        total_energy += magnitudes[i];
    }
    
    float threshold = total_energy * rolloff_percent;
    float cumulative = 0.0f;
    
    *rolloff_bin = 0;
    for (int i = 0; i < n_bins; i++) {
        cumulative += magnitudes[i];
        if (cumulative >= threshold) {
            *rolloff_bin = i;
            break;
        }
    }
    
    return 0;
}

int mve_spectral_flatness(const float* magnitudes, int n_bins, float* flatness) {
    if (!magnitudes || n_bins <= 0 || !flatness) return -1;
    
    float geometric_mean = 0.0f;
    float arithmetic_mean = 0.0f;
    
    for (int i = 0; i < n_bins; i++) {
        float mag = fmaxf(magnitudes[i], 1e-10f);
        geometric_mean += logf(mag);
        arithmetic_mean += magnitudes[i];
    }
    
    geometric_mean = expf(geometric_mean / n_bins);
    arithmetic_mean /= n_bins;
    
    *flatness = (arithmetic_mean > 1e-6f) ? (geometric_mean / arithmetic_mean) : 0.0f;
    return 0;
}

int mve_spectral_contrast(const float* magnitudes, int n_bins, int n_bands,
                          float* contrast) {
    /* Placeholder for spectral contrast */
    (void)magnitudes; (void)n_bins; (void)n_bands; (void)contrast;
    return 0;
}

int mve_zcr(const float* audio, int n_samples, int frame_size, int hop_size,
            float* zcr, int* n_frames) {
    if (!audio || !zcr || !n_frames || n_samples <= 0 || frame_size <= 0 || hop_size <= 0) {
        return -1;
    }
    
    *n_frames = 1 + (n_samples - frame_size) / hop_size;
    
    for (int frame = 0; frame < *n_frames; frame++) {
        int start = frame * hop_size;
        int crossings = 0;
        
        for (int i = 1; i < frame_size && (start + i) < n_samples; i++) {
            if ((audio[start + i - 1] >= 0 && audio[start + i] < 0) ||
                (audio[start + i - 1] < 0 && audio[start + i] >= 0)) {
                crossings++;
            }
        }
        
        zcr[frame] = (float)crossings / frame_size;
    }
    
    return 0;
}

/* ============================================================================
 * Multi-Resolution STFT Loss
 * ============================================================================
 */

MVE_STFTLossConfig* mve_stft_loss_config_create(int fft_size, int hop_size, int win_length,
                                                 MVE_WindowType window_type) {
    if (!mve_is_power_of_2(fft_size)) return NULL;
    
    MVE_STFTLossConfig* config = (MVE_STFTLossConfig*)calloc(1, sizeof(MVE_STFTLossConfig));
    if (!config) return NULL;
    
    config->fft_plan = mve_fft_create(fft_size);
    if (!config->fft_plan) {
        free(config);
        return NULL;
    }
    
    config->window = (float*)malloc(win_length * sizeof(float));
    config->log_mag_buffer = (float*)malloc((fft_size/2+1) * sizeof(float));
    config->lin_mag_buffer = (float*)malloc((fft_size/2+1) * sizeof(float));
    
    if (!config->window || !config->log_mag_buffer || !config->lin_mag_buffer) {
        mve_stft_loss_config_destroy(config);
        return NULL;
    }
    
    mve_window_generate(config->window, win_length, window_type, 0.0);
    
    config->fft_size = fft_size;
    config->hop_size = hop_size;
    config->win_length = win_length;
    config->window_type = window_type;
    
    return config;
}

void mve_stft_loss_config_destroy(MVE_STFTLossConfig* config) {
    if (!config) return;
    mve_fft_destroy(config->fft_plan);
    free(config->window);
    free(config->log_mag_buffer);
    free(config->lin_mag_buffer);
    free(config);
}

float mve_multi_res_stft_loss(const float* pred, const float* target, int n_samples,
                              int sample_rate, const int* fft_sizes, const int* hop_sizes,
                              const int* win_lengths, int n_resolutions,
                              float w_log_mag, float w_lin_mag) {
    if (!pred || !target || !fft_sizes || !hop_sizes || !win_lengths || n_samples <= 0) {
        return 0.0f;
    }
    
    float total_loss = 0.0f;
    
    for (int r = 0; r < n_resolutions; r++) {
        int fft_size = fft_sizes[r];
        int hop_size = hop_sizes[r];
        int win_length = win_lengths[r];
        
        MVE_STFTLossConfig* config = mve_stft_loss_config_create(fft_size, hop_size, 
                                                                  win_length, MVE_WINDOW_HANN);
        if (!config) continue;
        
        int num_bins = fft_size / 2 + 1;
        int num_frames = 1 + (n_samples - win_length) / hop_size;
        
        if (num_frames <= 0) {
            mve_stft_loss_config_destroy(config);
            continue;
        }
        
        float* pred_re = (float*)calloc(num_frames * num_bins, sizeof(float));
        float* pred_im = (float*)calloc(num_frames * num_bins, sizeof(float));
        float* target_re = (float*)calloc(num_frames * num_bins, sizeof(float));
        float* target_im = (float*)calloc(num_frames * num_bins, sizeof(float));
        
        if (!pred_re || !pred_im || !target_re || !target_im) {
            free(pred_re); free(pred_im); free(target_re); free(target_im);
            mve_stft_loss_config_destroy(config);
            continue;
        }
        
        /* Create temporary STFT config for forward pass */
        MVE_STFTConfig stft_config;
        stft_config.fft_size = fft_size;
        stft_config.hop_size = hop_size;
        stft_config.win_length = win_length;
        stft_config.num_freq_bins = num_bins;
        stft_config.window_type = MVE_WINDOW_HANN;
        stft_config.fft_plan = config->fft_plan;
        stft_config.window = config->window;
        stft_config.frame_buffer = (float*)calloc(fft_size, sizeof(float));
        stft_config.overlap_buffer = NULL;
        stft_config.window_sum = NULL;
        stft_config.num_frames = 0;
        
        if (!stft_config.frame_buffer) {
            free(pred_re); free(pred_im); free(target_re); free(target_im);
            mve_stft_loss_config_destroy(config);
            continue;
        }
        
        mve_stft_forward(&stft_config, pred, n_samples, pred_re, pred_im);
        mve_stft_forward(&stft_config, target, n_samples, target_re, target_im);
        
        free(stft_config.frame_buffer);
        
        float loss = 0.0f;
        for (int f = 0; f < num_frames; f++) {
            for (int b = 0; b < num_bins; b++) {
                int idx = f * num_bins + b;
                
                /* Log magnitude loss */
                float pred_log = logf(sqrtf(pred_re[idx]*pred_re[idx] + pred_im[idx]*pred_im[idx]) + 1e-6f);
                float target_log = logf(sqrtf(target_re[idx]*target_re[idx] + target_im[idx]*target_im[idx]) + 1e-6f);
                loss += w_log_mag * fabsf(pred_log - target_log);
                
                /* Linear magnitude loss */
                float pred_lin = sqrtf(pred_re[idx]*pred_re[idx] + pred_im[idx]*pred_im[idx]);
                float target_lin = sqrtf(target_re[idx]*target_re[idx] + target_im[idx]*target_im[idx]);
                loss += w_lin_mag * fabsf(pred_lin - target_lin);
            }
        }
        
        loss /= (num_frames * num_bins);
        total_loss += loss;
        
        free(pred_re); free(pred_im); free(target_re); free(target_im);
        mve_stft_loss_config_destroy(config);
    }
    
    return (n_resolutions > 0) ? (total_loss / n_resolutions) : 0.0f;
}

/* ============================================================================
 * Gradient Helpers
 * ============================================================================
 */

int mve_grad_logmag(const float* mag, const float* grad_logmag, float* grad_mag, 
                    int n, float eps) {
    if (!mag || !grad_logmag || !grad_mag || n <= 0) return -1;
    
    for (int i = 0; i < n; i++) {
        grad_mag[i] = grad_logmag[i] / fmaxf(mag[i], eps);
    }
    
    return 0;
}

int mve_grad_mel_filterbank(const MVE_MelFilterbank* fb, const float* grad_mel,
                            int n_frames, float* grad_spectrogram) {
    if (!fb || !grad_mel || !grad_spectrogram || n_frames <= 0) return -1;
    
    int num_bins = fb->n_fft / 2 + 1;
    memset(grad_spectrogram, 0, n_frames * num_bins * sizeof(float));
    
    for (int frame = 0; frame < n_frames; frame++) {
        for (int i = 0; i < fb->n_mels; i++) {
            for (int bin = fb->filter_starts[i]; bin <= fb->filter_ends[i]; bin++) {
                grad_spectrogram[frame * num_bins + bin] += 
                    grad_mel[frame * fb->n_mels + i] * fb->filterbank[i * num_bins + bin];
            }
        }
    }
    
    return 0;
}

int mve_grad_power_spectrum(const float* re, const float* im, const float* grad_power,
                            float* grad_re, float* grad_im, int n) {
    if (!re || !im || !grad_power || !grad_re || !grad_im || n <= 0) return -1;
    
    for (int i = 0; i < n; i++) {
        grad_re[i] = 2.0f * re[i] * grad_power[i];
        grad_im[i] = 2.0f * im[i] * grad_power[i];
    }
    
    return 0;
}

/* ============================================================================
 * Debug Utilities
 * ============================================================================
 */

void mve_print_spectrogram(const float* spec, int n_frames, int n_bins, const char* label) {
    if (!spec || !label) return;
    
    printf("Spectrogram: %s [%d frames x %d bins]\n", label, n_frames, n_bins);
    for (int f = 0; f < n_frames && f < 10; f++) {
        printf("  Frame %d:", f);
        for (int b = 0; b < n_bins && b < 10; b++) {
            printf(" %6.3f", spec[f * n_bins + b]);
        }
        printf("\n");
    }
}

/* Numerically stable log-sum-exp */
float mve_logsumexp(const float* x, int n) {
    if (n <= 0) return -INFINITY;
    
    /* Find maximum value */
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    
    /* If all values are -inf, return -inf */
    if (max_val == -INFINITY) return -INFINITY;
    
    /* Compute sum of exp(x_i - max) */
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += expf(x[i] - max_val);
    }
    
    return max_val + logf(sum);
}
