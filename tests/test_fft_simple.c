#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mve_dsp.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

int main(void) {
    printf("Testing simple FFT...\n");
    
    /* Use minimum supported size (64) since smaller sizes are rejected */
    int n = 64;
    MVE_FFTPlan* plan = mve_fft_create(n);
    if (!plan) {
        printf("FAILED: Could not create FFT plan for size %d\n", n);
        return 1;
    }
    
    /* Create delta function input (impulse at start) */
    float* re = (float*)calloc(n, sizeof(float));
    float* im = (float*)calloc(n, sizeof(float));
    re[0] = 1.0f;
    
    printf("Input: delta function (1 followed by %d zeros)\n", n-1);

    mve_fft_forward(plan, re, im);

    printf("FFT result - first 8 bins (re): ");
    for (int i = 0; i < 8 && i < n; i++) printf("%.4f ", re[i]);
    printf("...\n");

    /* Delta function FFT should be all 1s in real part, 0 in imaginary */
    float max_error = 0;
    for (int i = 0; i < n; i++) {
        float err_re = fabsf(re[i] - 1.0f);
        float err_im = fabsf(im[i]);
        if (err_re > max_error) max_error = err_re;
        if (err_im > max_error) max_error = err_im;
    }

    if (max_error < 1e-5f) {
        printf("PASSED: Delta function FFT correct (max error: %e)\n", max_error);
    } else {
        printf("FAILED: Max error = %f\n", max_error);
    }
    
    free(re);
    free(im);
    mve_fft_destroy(plan);
    return 0;
}
