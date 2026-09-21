#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mve_dsp.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

int main(void) {
    printf("Testing FFT roundtrip with sine wave...\n");
    
    int n = 256;
    MVE_FFTPlan* plan = mve_fft_create(n);
    if (!plan) {
        printf("FAILED: Could not create FFT plan\n");
        return 1;
    }
    
    float* re_in = (float*)calloc(n, sizeof(float));
    float* im_in = (float*)calloc(n, sizeof(float));
    float* re_out = (float*)calloc(n, sizeof(float));
    float* im_out = (float*)calloc(n, sizeof(float));
    
    /* Create test signal: pure sinusoid at bin 8 */
    for (int i = 0; i < n; i++) {
        re_in[i] = sinf(2.0f * M_PI * 8.0f * i / n);
        im_in[i] = 0.0f;
    }
    
    memcpy(re_out, re_in, n * sizeof(float));
    memcpy(im_out, im_in, n * sizeof(float));
    
    /* Forward FFT */
    mve_fft_forward(plan, re_out, im_out);
    
    printf("After forward FFT:\n");
    printf("  Max re: %.4f, Max im: %.4f\n", fabsf(re_out[0]), fabsf(im_out[0]));
    
    /* Inverse FFT */
    mve_fft_inverse(plan, re_out, im_out);
    
    /* Check reconstruction */
    float max_error = 0.0f;
    for (int i = 0; i < n; i++) {
        float error = fabsf(re_out[i] - re_in[i]);
        if (error > max_error) max_error = error;
    }
    
    printf("Roundtrip max error: %e\n", max_error);
    
    if (max_error < 1e-4f) {
        printf("PASSED\n");
    } else {
        printf("FAILED\n");
        /* Print first few samples */
        printf("Original: ");
        for (int i = 0; i < 8; i++) printf("%.4f ", re_in[i]);
        printf("\nReconstructed: ");
        for (int i = 0; i < 8; i++) printf("%.4f ", re_out[i]);
        printf("\n");
    }
    
    free(re_in); free(im_in); free(re_out); free(im_out);
    mve_fft_destroy(plan);
    return 0;
}
