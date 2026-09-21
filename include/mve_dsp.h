#ifndef MVE_DSP_H
#define MVE_DSP_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * MVE-1: First-Party DSP Engine
 * High-Performance Audio Processing with Numerical Stability
 * 
 * Algorithmic Features:
 * - Mixed-Radix FFT (Factors 2, 3, 4, 5) + Bluestein for Primes
 * - YIN Algorithm for Robust F0 Estimation  
 * - Kahan Summation for Accumulation Stability
 * - Log-Sum-Exp for Stable Softmax/Loss Computation
 * - Polyphase Resampling with Kaiser Windows
 * ============================================================================ */

#define MVE_DSP_MAX_FFT_SIZE 8192
#define MVE_DSP_MIN_FFT_SIZE 16
#define MVE_DSP_MAX_MEL_CHANNELS 256
#define MVE_DSP_MAX_WINDOW_SIZE 8192
#define MVE_DSP_YIN_THRESHOLD 0.15f
#define MVE_DSP_KAHAN_EPSILON 1e-8f
#define MVE_DSP_LOG_EPSILON 1e-10f

/* ============================================================================
 * Window Functions
 * ============================================================================ */

typedef enum {
    MVE_WINDOW_RECTANGULAR = 0,
    MVE_WINDOW_HANN,
    MVE_WINDOW_HAMMING,
    MVE_WINDOW_BLACKMAN,
    MVE_WINDOW_BLACKMAN_HARRIS,
    MVE_WINDOW_NUTTALL,
    MVE_WINDOW_KAISER
} MVE_WindowType;

/* Generate window function in-place */
int mve_window_generate(float* window, int size, MVE_WindowType type, double param);

/* Apply window to buffer */
int mve_window_apply(float* buffer, const float* window, int size);

/* ============================================================================
 * FFT Engine (Mixed-Radix Cooley-Tukey + Bluestein)
 * ============================================================================ */

typedef struct MVE_FFTPlan_s {
    int n;                  /* FFT size */
    int log2n;              /* log2(n) for power-of-2 sizes */
    bool is_power_of_2;     /* Optimization flag */
    int num_factors;        /* Number of prime factors */
    int factors[16];        /* Prime factorization */
    
    /* Twiddle factors */
    float* cos_table;       /* Precomputed cos values */
    float* sin_table;       /* Precomputed sin values */
    int* bit_reverse_table; /* Bit-reversal permutation */
    bool owns_tables;       /* Whether this struct owns the tables */
    
    /* Bluestein chirp z-transform buffers for prime sizes (forward declared) */
    struct MVE_FFTPlan_s* bluestein_fft_plan;  /* Nested plan for Bluestein */
} MVE_FFTPlan;

/* Create FFT plan for given size (supports any size via mixed-radix + Bluestein) */
MVE_FFTPlan* mve_fft_create(int n);
void mve_fft_destroy(MVE_FFTPlan* plan);

/* Forward FFT: complex input -> complex output (in-place allowed) */
int mve_fft_forward(MVE_FFTPlan* plan, float* re, float* im);

/* Inverse FFT: complex input -> complex output (in-place allowed) */
int mve_fft_inverse(MVE_FFTPlan* plan, float* re, float* im);

/* Real FFT: real input -> packed complex output (optimized) */
int mve_rfft_forward(MVE_FFTPlan* plan, float* input, float* output);

/* Real inverse FFT: packed complex -> real output (optimized) */
int mve_rfft_inverse(MVE_FFTPlan* plan, float* input, float* output);

/* Power spectrum from real FFT output */
int mve_power_spectrum(const float* re, const float* im, int n, float* power);

/* Magnitude spectrum */
int mve_magnitude_spectrum(const float* re, const float* im, int n, float* mag);

/* Log-magnitude spectrum with numerical stability */
int mve_log_magnitude_spectrum(const float* re, const float* im, int n, float* log_mag, float eps);

/* ============================================================================
 * Numerical Stability Utilities
 * ============================================================================ */

/* Dot product with Kahan summation for reduced floating point error */
float mve_dot_kahan(const float* a, const float* b, int n);

/* Log-Sum-Exp trick: log(sum(exp(x))) - prevents overflow/underflow */
float mve_logsumexp(const float* x, int n);

/* Stable softmax computation */
void mve_softmax_stable(const float* x, int n, float* out);

/* Next power of 2 >= n (bit-twiddling) */
int mve_next_power_of_2(int n);

/* Check if number is power of 2 */
int mve_is_power_of_2(int n);

/* Integer log2 */
int mve_log2_int(int n);

/* Greatest common divisor */
int mve_gcd(int a, int b);

/* Least common multiple */
int mve_lcm(int a, int b);

/* ============================================================================
 * STFT Engine
 * ============================================================================ */

typedef struct {
    int fft_size;
    int hop_size;
    int win_length;
    MVE_WindowType window_type;
    float* window;
    MVE_FFTPlan* fft_plan;
    int num_frames;
    int num_freq_bins;
    
    /* Scratch buffers */
    float* frame_buffer;
    float* overlap_buffer;
    float* window_sum;
} MVE_STFTConfig;

/* Initialize STFT configuration */
MVE_STFTConfig* mve_stft_config_create(int fft_size, int hop_size, int win_length, 
                                        MVE_WindowType window_type);
void mve_stft_config_destroy(MVE_STFTConfig* config);

/* Forward STFT: time domain -> complex spectrogram */
int mve_stft_forward(
    MVE_STFTConfig* config,
    const float* input,
    int input_samples,
    float* spectrogram_re,  /* [num_frames][num_freq_bins] */
    float* spectrogram_im   /* [num_frames][num_freq_bins] */
);

/* Inverse STFT: complex spectrogram -> time domain */
int mve_stft_inverse(
    MVE_STFTConfig* config,
    const float* spectrogram_re,
    const float* spectrogram_im,
    int num_frames,
    float* output,
    int output_samples
);

/* Gradients for STFT (for backprop) */
int mve_stft_backward(
    MVE_STFTConfig* config,
    const float* grad_spectrogram_re,
    const float* grad_spectrogram_im,
    int num_frames,
    float* grad_input
);

/* ============================================================================
 * Mel Filterbank
 * ============================================================================ */

typedef struct {
    int sample_rate;
    int n_fft;
    int n_mels;
    float fmin;
    float fmax;
    float* filterbank;      /* [n_mels][n_fft/2+1] */
    int* filter_starts;     /* Start index for each mel filter */
    int* filter_ends;       /* End index for each mel filter */
} MVE_MelFilterbank;

/* Create mel filterbank */
MVE_MelFilterbank* mve_mel_filterbank_create(int sample_rate, int n_fft, int n_mels,
                                              float fmin, float fmax);
void mve_mel_filterbank_destroy(MVE_MelFilterbank* fb);

/* Convert linear spectrogram to mel spectrogram */
int mve_spectrogram_to_mel(
    const MVE_MelFilterbank* fb,
    const float* spectrogram,  /* [n_frames][n_fft/2+1] */
    int n_frames,
    float* mel_spec            /* [n_frames][n_mels] */
);

/* Convert mel spectrogram to log-mel */
int mve_mel_to_logmel(float* mel_spec, int n_frames, int n_mels, float eps);

/* ============================================================================
 * F0 Extraction (YIN Algorithm with Parabolic Interpolation)
 * ============================================================================ */

typedef struct {
    int sample_rate;
    int frame_size;
    int hop_size;
    float f0_min;
    float f0_max;
    float voicing_threshold;
    
    /* Internal buffers for YIN algorithm */
    float* diff_buffer;       /* Difference function */
    float* cumsum_buffer;     /* Cumulative mean normalized difference */
    float* parabolic_buffer;  /* For sub-sample interpolation */
    int min_lag;              /* Corresponds to f0_max */
    int max_lag;              /* Corresponds to f0_min */
} MVE_F0Extractor;

/* Create F0 extractor with YIN algorithm */
MVE_F0Extractor* mve_f0_extractor_create(int sample_rate, int frame_size, int hop_size,
                                          float f0_min, float f0_max);
void mve_f0_extractor_destroy(MVE_F0Extractor* ext);

/* Extract F0 contour using YIN algorithm */
int mve_f0_extract(
    MVE_F0Extractor* ext,
    const float* audio,
    int n_samples,
    float* f0_contour,      /* F0 in Hz, 0 for unvoiced */
    float* voicing_prob,    /* Probability of voicing [0, 1] */
    int* n_frames
);

/* Extract F0 with sub-sample precision via parabolic interpolation */
int mve_f0_extract_precise(
    MVE_F0Extractor* ext,
    const float* audio,
    int n_samples,
    float* f0_contour,      /* F0 in Hz */
    float* lag_samples,     /* Exact lag with fractional part */
    float* confidence,      /* Confidence score [0, 1] */
    bool* voiced,           /* Voiced/unvoiced decision */
    int* n_frames
);

/* Alternative: PYIN-style probabilistic F0 with Viterbi decoding */
int mve_f0_extract_pyin(
    MVE_F0Extractor* ext,
    const float* audio,
    int n_samples,
    float* f0_contour,
    float* voicing_prob,
    int* n_frames
);

/* ============================================================================
 * Energy Extraction
 * ============================================================================ */

/* RMS energy per frame */
int mve_energy_rms(
    const float* audio,
    int n_samples,
    int frame_size,
    int hop_size,
    float* energy,
    int* n_frames
);

/* dB energy per frame */
int mve_energy_db(
    const float* audio,
    int n_samples,
    int frame_size,
    int hop_size,
    float* energy_db,
    int* n_frames,
    float ref_level
);

/* ============================================================================
 * Resampling
 * ============================================================================ */

typedef struct {
    int input_rate;
    int output_rate;
    int lFactor;              /* LCM-based upsample factor */
    int mFactor;              /* LCM-based downsample factor */
    float* polyphase_filters; /* Polyphase filter coefficients */
    int filter_order;
    int num_phases;
    
    /* State for streaming */
    float* state_buffer;
    int state_index;
} MVE_Resampler;

/* Create resampler using polyphase FIR */
MVE_Resampler* mve_resampler_create(int input_rate, int output_rate, int filter_order);
void mve_resampler_destroy(MVE_Resampler* res);

/* Resample entire buffer */
int mve_resample(
    MVE_Resampler* res,
    const float* input,
    int n_input,
    float* output,
    int* n_output
);

/* Streaming resample (for real-time) */
int mve_resample_stream(
    MVE_Resampler* res,
    const float* input,
    int n_input,
    float* output,
    int max_output,
    int* n_output,
    int* end_of_stream
);

/* ============================================================================
 * Silence Detection
 * ============================================================================ */

typedef struct {
    float threshold_db;
    int min_silence_frames;
    int min_speech_frames;
    int frame_size;
    int hop_size;
} MVE_SilenceDetector;

/* Detect silence regions */
int mve_silence_detect(
    MVE_SilenceDetector* det,
    const float* audio,
    int n_samples,
    int* silence_starts,    /* Sample indices where silence begins */
    int* silence_ends,      /* Sample indices where silence ends */
    int* n_regions,
    int max_regions
);

/* Trim silence from start and end */
int mve_silence_trim(
    MVE_SilenceDetector* det,
    const float* audio,
    int n_samples,
    int* start_sample,
    int* end_sample
);

/* ============================================================================
 * Normalization
 * ============================================================================ */

/* Peak normalization */
int mve_normalize_peak(float* audio, int n_samples, float target_peak);

/* RMS normalization */
int mve_normalize_rms(float* audio, int n_samples, float target_rms);

/* Loudness normalization (simplified EBU R128-ish) */
int mve_normalize_loudness(float* audio, int n_samples, int sample_rate, float target_lufs);

/* ============================================================================
 * Spectral Features
 * ============================================================================ */

/* Spectral centroid */
int mve_spectral_centroid(const float* magnitudes, int n_bins, float* centroid);

/* Spectral bandwidth */
int mve_spectral_bandwidth(const float* magnitudes, int n_bins, float centroid, float* bandwidth);

/* Spectral rolloff */
int mve_spectral_rolloff(const float* magnitudes, int n_bins, float rolloff_percent, int* rolloff_bin);

/* Spectral flatness */
int mve_spectral_flatness(const float* magnitudes, int n_bins, float* flatness);

/* Spectral contrast */
int mve_spectral_contrast(const float* magnitudes, int n_bins, int n_bands, float* contrast);

/* Zero crossing rate */
int mve_zcr(const float* audio, int n_samples, int frame_size, int hop_size, 
            float* zcr, int* n_frames);

/* ============================================================================
 * Multi-Resolution STFT Loss Support
 * ============================================================================ */

typedef struct {
    int fft_size;
    int hop_size;
    int win_length;
    MVE_WindowType window_type;
    MVE_FFTPlan* fft_plan;
    float* window;
    float* log_mag_buffer;
    float* lin_mag_buffer;
} MVE_STFTLossConfig;

/* Create STFT loss configuration */
MVE_STFTLossConfig* mve_stft_loss_config_create(int fft_size, int hop_size, int win_length,
                                                 MVE_WindowType window_type);
void mve_stft_loss_config_destroy(MVE_STFTLossConfig* config);

/* Compute multi-resolution STFT loss between two signals */
float mve_multi_res_stft_loss(
    const float* pred,
    const float* target,
    int n_samples,
    int sample_rate,
    const int* fft_sizes,         /* Array of FFT sizes */
    const int* hop_sizes,         /* Array of hop sizes */
    const int* win_lengths,       /* Array of window lengths */
    int n_resolutions,
    float w_log_mag,
    float w_lin_mag
);

/* ============================================================================
 * Gradient Helpers for DSP Operations
 * ============================================================================ */

/* Gradient through log-magnitude */
int mve_grad_logmag(const float* mag, const float* grad_logmag, float* grad_mag, int n, float eps);

/* Gradient through mel filterbank */
int mve_grad_mel_filterbank(
    const MVE_MelFilterbank* fb,
    const float* grad_mel,
    int n_frames,
    float* grad_spectrogram
);

/* Gradient through power spectrum */
int mve_grad_power_spectrum(
    const float* re, const float* im,
    const float* grad_power,
    float* grad_re, float* grad_im,
    int n
);

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

/* Check if number is power of 2 */
int mve_is_power_of_2(int n);

/* Next power of 2 >= n */
int mve_next_power_of_2(int n);

/* Integer log2 */
int mve_log2_int(int n);

/* Hertz to mel conversion */
float mve_hz_to_mel(float hz);

/* Mel to hertz conversion */
float mve_mel_to_hz(float mel);

/* Frequency to FFT bin */
int mve_freq_to_bin(float freq, int sample_rate, int fft_size);

/* FFT bin to frequency */
float mve_bin_to_freq(int bin, int sample_rate, int fft_size);

/* Print spectrogram (debug) */
void mve_print_spectrogram(const float* spec, int n_frames, int n_bins, const char* label);

#ifdef __cplusplus
}
#endif

#endif /* MVE_DSP_H */
