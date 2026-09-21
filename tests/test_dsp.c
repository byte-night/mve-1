#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mve_dsp.h"

#define EPSILON 1e-5f
#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

int test_fft_roundtrip(void) {
    printf("Testing FFT roundtrip...\n");
    
    int n = 256;
    MVE_FFTPlan* plan = mve_fft_create(n);
    if (!plan) {
        printf("  FAILED: Could not create FFT plan\n");
        return -1;
    }
    
    float* re_in = (float*)calloc(n, sizeof(float));
    float* im_in = (float*)calloc(n, sizeof(float));
    float* re_out = (float*)calloc(n, sizeof(float));
    float* im_out = (float*)calloc(n, sizeof(float));
    
    for (int i = 0; i < n; i++) {
        re_in[i] = sinf(2.0f * M_PI * 8.0f * i / n) + 0.5f * sinf(2.0f * M_PI * 32.0f * i / n);
        im_in[i] = 0.0f;
    }
    
    memcpy(re_out, re_in, n * sizeof(float));
    memcpy(im_out, im_in, n * sizeof(float));
    
    if (mve_fft_forward(plan, re_out, im_out) != 0) {
        printf("  FAILED: Forward FFT failed\n");
        goto cleanup;
    }
    
    if (mve_fft_inverse(plan, re_out, im_out) != 0) {
        printf("  FAILED: Inverse FFT failed\n");
        goto cleanup;
    }
    
    float max_error = 0.0f;
    for (int i = 0; i < n; i++) {
        float error = fabsf(re_out[i] - re_in[i]);
        if (error > max_error) max_error = error;
    }
    
    if (max_error < EPSILON) {
        printf("  PASSED: Max error = %e\n", max_error);
        free(re_in); free(im_in); free(re_out); free(im_out);
        mve_fft_destroy(plan);
        return 0;
    } else {
        printf("  FAILED: Max error = %e\n", max_error);
        goto cleanup;
    }
    
cleanup:
    free(re_in); free(im_in); free(re_out); free(im_out);
    mve_fft_destroy(plan);
    return -1;
}

int test_mel_filterbank(void) {
    printf("Testing Mel filterbank...\n");
    
    int sample_rate = 22050;
    int n_fft = 512;
    int n_mels = 80;
    
    MVE_MelFilterbank* fb = mve_mel_filterbank_create(sample_rate, n_fft, n_mels, 0.0f, sample_rate / 2.0f);
    if (!fb) {
        printf("  FAILED: Could not create mel filterbank\n");
        return -1;
    }
    
    if (fb->n_mels != n_mels) {
        printf("  FAILED: Expected %d mel channels, got %d\n", n_mels, fb->n_mels);
        mve_mel_filterbank_destroy(fb);
        return -1;
    }
    
    printf("  PASSED: Created %d mel filters\n", n_mels);
    mve_mel_filterbank_destroy(fb);
    return 0;
}

int main(void) {
    printf("=== MVE-1 DSP Unit Tests ===\n\n");
    
    int failures = 0;
    failures += test_fft_roundtrip();
    failures += test_mel_filterbank();
    
    printf("\n=== Results: %d failures ===\n", failures);
    return failures;
}
