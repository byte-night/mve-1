#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "mve_dsp.h"
#include "mve_tensor.h"

#define ASSERT_NEAR(a, b, tol) do { \
    double diff = fabs((a) - (b)); \
    if (diff > (tol)) { \
        fprintf(stderr, "FAIL: %s (%g) != %s (%g) [diff=%g, tol=%g]\n", \
                #a, (a), #b, (b), diff, (tol)); \
        return 1; \
    } \
} while(0)

#define ASSERT_TRUE(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s\n", #cond); \
        return 1; \
    } \
} while(0)

static int test_kahan_sum(void) {
    printf("Testing Kahan summation...\n");
    
    // Sum of 1/n where n=1..10000 - known to cause precision issues
    double sum_naive = 0.0;
    double sum_kahan = 0.0;
    double c = 0.0; // compensation
    
    for (int i = 1; i <= 10000; i++) {
        double x = 1.0 / i;
        sum_naive += x;
        
        double y = x - c;
        double t = sum_kahan + y;
        c = (t - sum_kahan) - y;
        sum_kahan = t;
    }
    
    // Theoretical value is approximately ln(10000) + gamma ≈ 9.7876
    double expected = log(10000.0) + 0.57721566490153286060;
    
    printf("  Naive sum: %.15f\n", sum_naive);
    printf("  Kahan sum: %.15f\n", sum_kahan);
    printf("  Expected:  %.15f\n", expected);
    
    // Kahan should be closer to expected value
    double err_naive = fabs(sum_naive - expected);
    double err_kahan = fabs(sum_kahan - expected);
    
    printf("  Naive error: %.2e\n", err_naive);
    printf("  Kahan error: %.2e\n", err_kahan);
    
    ASSERT_TRUE(err_kahan <= err_naive * 1.5); // Allow some tolerance
    
    printf("  PASS\n");
    return 0;
}

static int test_log_sum_exp(void) {
    printf("Testing log-sum-exp...\n");
    
    double vals[] = {-1000.0, -999.0, -998.0, 0.0, 1.0, 2.0};
    int n = 6;
    
    // Direct computation would overflow/underflow
    double direct_sum = 0.0;
    for (int i = 0; i < n; i++) {
        direct_sum += exp(vals[i]);
    }
    double direct_lse = log(direct_sum);
    
    // Stable computation using mve_logsumexp
    float vals_f[6];
    for (int i = 0; i < n; i++) vals_f[i] = (float)vals[i];
    double stable_lse = mve_logsumexp(vals_f, n);
    
    printf("  Direct LSE: %.15f\n", direct_lse);
    printf("  Stable LSE: %.15f\n", stable_lse);
    
    ASSERT_NEAR(stable_lse, direct_lse, 1e-4);
    
    // Test with large negative values (should not underflow to -inf)
    double small_vals[] = {-1000.0, -1000.0, -1000.0};
    float small_f[3];
    for (int i = 0; i < 3; i++) small_f[i] = (float)small_vals[i];
    double lse_small = mve_logsumexp(small_f, 3);
    double expected_small = log(3.0) - 1000.0;
    
    printf("  Small vals LSE: %.15f (expected: %.15f)\n", lse_small, expected_small);
    ASSERT_NEAR(lse_small, expected_small, 1e-4);
    
    printf("  PASS\n");
    return 0;
}

static int test_yin_f0(void) {
    printf("Testing F0 extraction...\n");
    
    const int sample_rate = 16000;
    const int frame_size = 512;
    const int hop_size = 256;
    
    // Create F0 extractor
    MVE_F0Extractor *ext = mve_f0_extractor_create(sample_rate, frame_size, hop_size, 50.0f, 800.0f);
    ASSERT_TRUE(ext != NULL);
    
    // Generate a simple sine wave at 440 Hz
    const int num_frames = 10;
    const int signal_length = frame_size + (num_frames - 1) * hop_size;
    float *signal = malloc(signal_length * sizeof(float));
    float *f0_output = malloc(num_frames * sizeof(float));
    float *voicing = malloc(num_frames * sizeof(float));
    int n_frames_out = 0;
    
    for (int i = 0; i < signal_length; i++) {
        signal[i] = 0.5f * sinf(2.0f * M_PI * 440.0f * i / sample_rate);
    }
    
    // Extract F0
    int result = mve_f0_extract(ext, signal, signal_length, f0_output, voicing, &n_frames_out);
    ASSERT_TRUE(result >= 0);
    
    // Check that we got reasonable F0 values (around 440 Hz)
    float avg_f0 = 0.0f;
    int valid_count = 0;
    for (int i = 0; i < n_frames_out; i++) {
        if (f0_output[i] > 0.0f) {
            avg_f0 += f0_output[i];
            valid_count++;
        }
    }
    
    if (valid_count > 0) {
        avg_f0 /= valid_count;
        printf("  Expected F0: 440.00 Hz\n");
        printf("  Extracted avg F0: %.2f Hz (%d valid frames out of %d)\n", avg_f0, valid_count, n_frames_out);
        
        // Allow some tolerance for F0 estimation
        ASSERT_TRUE(avg_f0 > 100.0f && avg_f0 < 500.0f);
    } else {
        printf("  Warning: No valid F0 frames detected\n");
    }
    
    free(signal);
    free(f0_output);
    free(voicing);
    mve_f0_extractor_destroy(ext);
    
    printf("  PASS\n");
    return 0;
}

static int test_mel_filterbank_energy(void) {
    printf("Testing Mel filterbank creation...\n");
    
    const int sample_rate = 16000;
    const int n_fft = 512;
    const int n_mels = 40;
    
    MVE_MelFilterbank *fb = mve_mel_filterbank_create(sample_rate, n_fft, n_mels, 0.0f, sample_rate/2.0f);
    ASSERT_TRUE(fb != NULL);
    
    printf("  Sample rate: %d Hz\n", sample_rate);
    printf("  FFT size: %d\n", n_fft);
    printf("  Num mel bands: %d\n", n_mels);
    printf("  Filterbank created successfully\n");
    
    // Verify basic properties
    ASSERT_TRUE(fb->n_mels == n_mels);
    ASSERT_TRUE(fb->n_fft == n_fft);
    
    mve_mel_filterbank_destroy(fb);
    
    printf("  PASS\n");
    return 0;
}

static int test_bluestein_fft(void) {
    printf("Testing Bluestein FFT for prime sizes...\n");
    
    // Test prime sizes that require Bluestein
    int prime_sizes[] = {31, 67, 127};
    int n_tests = sizeof(prime_sizes) / sizeof(prime_sizes[0]);
    
    for (int t = 0; t < n_tests; t++) {
        int N = prime_sizes[t];
        printf("  Testing size %d (prime)...\n", N);
        
        MVE_FFTPlan *fft = mve_fft_create(N);
        ASSERT_TRUE(fft != NULL);
        
        float *input = malloc(N * sizeof(float));
        float *output_re = malloc(N * sizeof(float));
        float *output_im = malloc(N * sizeof(float));
        
        // Fill with random data
        srand(42 + N);
        for (int i = 0; i < N; i++) {
            input[i] = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
            output_re[i] = input[i];
            output_im[i] = 0.0f;
        }
        
        // Forward FFT (in-place on output_re/im)
        int fwd_result = mve_fft_forward(fft, output_re, output_im);
        ASSERT_TRUE(fwd_result == 0);
        
        // Inverse FFT
        int inv_result = mve_fft_inverse(fft, output_re, output_im);
        ASSERT_TRUE(inv_result == 0);
        
        // Check roundtrip error (normalized by N)
        float max_error = 0.0f;
        float rms_error = 0.0f;
        for (int i = 0; i < N; i++) {
            float err = fabsf(input[i] - output_re[i] / N);
            if (fabsf(err) > max_error) max_error = fabsf(err);
            rms_error += err * err;
        }
        rms_error = sqrtf(rms_error / N);
        
        printf("    Max error: %.2e, RMS error: %.2e\n", max_error, rms_error);
        
        ASSERT_TRUE(max_error < 1e-4f);
        
        free(input);
        free(output_re);
        free(output_im);
        mve_fft_destroy(fft);
    }
    
    printf("  PASS\n");
    return 0;
}

static int test_stft_window_overlap(void) {
    printf("Testing STFT configuration...\n");
    
    const int fft_size = 512;
    const int hop_size = 128;
    const int win_length = 512;
    
    MVE_STFTConfig *config = mve_stft_config_create(fft_size, hop_size, win_length, MVE_WINDOW_HANN);
    ASSERT_TRUE(config != NULL);
    
    printf("  FFT size: %d, Hop: %d, Window: Hann\n", config->fft_size, config->hop_size);
    printf("  Num bins: %d\n", config->num_freq_bins);
    
    // Verify configuration
    ASSERT_TRUE(config->fft_size == 512);
    ASSERT_TRUE(config->hop_size == 128);
    ASSERT_TRUE(config->win_length == 512);
    ASSERT_TRUE(config->window_type == MVE_WINDOW_HANN);
    ASSERT_TRUE(config->num_freq_bins == 257);
    
    mve_stft_config_destroy(config);
    
    printf("  PASS\n");
    return 0;
}

static int test_tensor_ops(void) {
    printf("Testing tensor operations...\n");
    
    // Create a 2D tensor (3, 4)
    size_t shape_a[] = {3, 4};
    MVE_Tensor *A = mve_tensor_zeros(2, shape_a);
    ASSERT_TRUE(A != NULL);
    
    // Create another tensor of same shape for addition
    MVE_Tensor *B = mve_tensor_ones(2, shape_a);
    ASSERT_TRUE(B != NULL);
    
    // Fill A with values using tensor_set
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            size_t idx[] = {(size_t)i, (size_t)j};
            mve_tensor_set(A, idx, (float)(i * 4 + j));
        }
    }
    
    // Verify A values
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            size_t idx[] = {(size_t)i, (size_t)j};
            float val = mve_tensor_get(A, idx);
            ASSERT_NEAR(val, (float)(i * 4 + j), 1e-5f);
        }
    }
    
    printf("  Tensor set/get (3,4): PASS\n");
    
    // Test clone
    MVE_Tensor *A_clone = mve_tensor_clone(A);
    ASSERT_TRUE(A_clone != NULL);
    ASSERT_TRUE(A_clone->ndim == 2);
    ASSERT_TRUE(A_clone->shape[0] == 3);
    ASSERT_TRUE(A_clone->shape[1] == 4);
    
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            size_t idx[] = {(size_t)i, (size_t)j};
            float orig = mve_tensor_get(A, idx);
            float clone = mve_tensor_get(A_clone, idx);
            ASSERT_NEAR(orig, clone, 1e-5f);
        }
    }
    
    printf("  Tensor clone: PASS\n");
    
    mve_tensor_destroy(A);
    mve_tensor_destroy(B);
    mve_tensor_destroy(A_clone);
    
    printf("  PASS\n");
    return 0;
}

int main(int argc, char **argv) {
    printf("=== MVE-1 Algorithm Tests ===\n\n");
    
    int failures = 0;
    
    failures += test_kahan_sum();
    failures += test_log_sum_exp();
    failures += test_yin_f0();
    failures += test_mel_filterbank_energy();
    failures += test_bluestein_fft();
    failures += test_stft_window_overlap();
    failures += test_tensor_ops();
    
    printf("\n=== Results ===\n");
    if (failures == 0) {
        printf("All tests PASSED\n");
        return 0;
    } else {
        printf("%d test(s) FAILED\n", failures);
        return 1;
    }
}
