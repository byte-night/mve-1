#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mve_dsp.h"

int main(void) {
    printf("Testing FFT size 8 (min is 64)...\n");
    
    /* Test with valid sizes */
    int sizes[] = {64, 128, 256, 512};
    for (int s = 0; s < 4; s++) {
        int n = sizes[s];
        MVE_FFTPlan* plan = mve_fft_create(n);
        if (plan) {
            printf("Size %d: OK\n", n);
            mve_fft_destroy(plan);
        } else {
            printf("Size %d: FAILED to create\n", n);
        }
    }
    
    return 0;
}
