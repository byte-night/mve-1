/* ============================================================================
 * MVE-1: Monologue Voice Engine - WAV File I/O Implementation
 * 
 * First-party RIFF/WAV reader and writer.
 * No external libraries (libsndfile, ffmpeg, etc.)
 * ============================================================================
 */

#include "mve.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* WAV header structures */
typedef struct {
    char riff_id[4];      /* "RIFF" */
    uint32_t file_size;   /* File size - 8 */
    char wave_id[4];      /* "WAVE" */
} WavHeader_RIFF;

typedef struct {
    char fmt_id[4];       /* "fmt " */
    uint32_t chunk_size;  /* 16 for PCM */
    uint16_t audio_fmt;   /* 1 = PCM, 3 = IEEE float */
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
} WavHeader_FMT;

typedef struct {
    char data_id[4];      /* "data" */
    uint32_t data_size;   /* Byte count of samples */
} WavHeader_DATA;

/* Read WAV file - outputs float samples in range [-1, 1] */
MVE_Error mve_wav_read(const char* path, float** samples, size_t* num_samples, int* sample_rate) {
    if (!path || !samples || !num_samples || !sample_rate) {
        return MVE_ERR_INVALID_ARG;
    }
    
    FILE* f = fopen(path, "rb");
    if (!f) {
        return MVE_ERR_FILE_NOT_FOUND;
    }
    
    /* Read RIFF header */
    WavHeader_RIFF riff;
    if (fread(&riff, sizeof(riff), 1, f) != 1) {
        fclose(f);
        return MVE_ERR_FILE_FORMAT;
    }
    
    /* Validate RIFF */
    if (memcmp(riff.riff_id, "RIFF", 4) != 0 || memcmp(riff.wave_id, "WAVE", 4) != 0) {
        fclose(f);
        return MVE_ERR_FILE_FORMAT;
    }
    
    /* Find fmt chunk */
    WavHeader_FMT fmt;
    char chunk_id[4];
    uint32_t chunk_size;
    
    int found_fmt = 0;
    while (fread(chunk_id, 4, 1, f) == 1 && fread(&chunk_size, 4, 1, f) == 1) {
        if (memcmp(chunk_id, "fmt ", 4) == 0) {
            if (chunk_size < sizeof(fmt) - 8) {
                fclose(f);
                return MVE_ERR_FILE_FORMAT;
            }
            if (fread(&fmt.audio_fmt, chunk_size, 1, f) != 1) {
                fclose(f);
                return MVE_ERR_FILE_FORMAT;
            }
            found_fmt = 1;
            break;
        } else {
            /* Skip unknown chunk */
            fseek(f, chunk_size, SEEK_CUR);
        }
    }
    
    if (!found_fmt) {
        fclose(f);
        return MVE_ERR_FILE_FORMAT;
    }
    
    /* Validate format */
    if (fmt.audio_fmt != 1 && fmt.audio_fmt != 3) {  /* PCM or IEEE float */
        fclose(f);
        return MVE_ERR_FILE_FORMAT;
    }
    
    *sample_rate = fmt.sample_rate;
    
    /* Find data chunk */
    WavHeader_DATA data;
    int found_data = 0;
    
    while (fread(chunk_id, 4, 1, f) == 1 && fread(&chunk_size, 4, 1, f) == 1) {
        if (memcmp(chunk_id, "data", 4) == 0) {
            data.data_size = chunk_size;
            found_data = 1;
            break;
        } else {
            fseek(f, chunk_size, SEEK_CUR);
        }
    }
    
    if (!found_data) {
        fclose(f);
        return MVE_ERR_FILE_FORMAT;
    }
    
    /* Calculate sample count */
    uint16_t bytes_per_sample = fmt.bits_per_sample / 8;
    uint16_t channels = fmt.num_channels;
    size_t total_samples = data.data_size / (bytes_per_sample * channels);
    
    /* Allocate output buffer */
    float* buf = (float*)calloc(total_samples, sizeof(float));
    if (!buf) {
        fclose(f);
        return MVE_ERR_OUT_OF_MEMORY;
    }
    
    /* Read and convert samples */
    for (size_t i = 0; i < total_samples; i++) {
        double sample = 0.0;
        
        if (fmt.audio_fmt == 1) {
            /* PCM */
            if (fmt.bits_per_sample == 16) {
                int16_t s;
                if (fread(&s, 2, 1, f) != 1) break;
                sample = s / 32768.0;
            } else if (fmt.bits_per_sample == 24) {
                uint8_t b[3];
                if (fread(b, 3, 1, f) != 1) break;
                int32_t s = (b[0] | (b[1] << 8) | (b[2] << 16));
                if (s & 0x800000) s |= 0xFF000000;  /* Sign extend */
                sample = s / 8388608.0;
            } else if (fmt.bits_per_sample == 32) {
                int32_t s;
                if (fread(&s, 4, 1, f) != 1) break;
                sample = s / 2147483648.0;
            }
        } else if (fmt.audio_fmt == 3) {
            /* IEEE float */
            if (fmt.bits_per_sample == 32) {
                float s;
                if (fread(&s, 4, 1, f) != 1) break;
                sample = s;
            } else if (fmt.bits_per_sample == 64) {
                double s;
                if (fread(&s, 8, 1, f) != 1) break;
                sample = s;
            }
        }
        
        /* Sum channels to mono */
        buf[i] += (float)sample;
        
        /* Skip remaining channels */
        for (int c = 1; c < channels; c++) {
            fseek(f, bytes_per_sample, SEEK_CUR);
        }
    }
    
    fclose(f);
    
    *samples = buf;
    *num_samples = total_samples;
    
    return MVE_OK;
}

/* Write WAV file - expects float samples in range [-1, 1] */
MVE_Error mve_wav_write(const char* path, const float* samples, size_t num_samples, int sample_rate) {
    if (!path || !samples || num_samples == 0 || sample_rate <= 0) {
        return MVE_ERR_INVALID_ARG;
    }
    
    FILE* f = fopen(path, "wb");
    if (!f) {
        return MVE_ERR_FILE_NOT_FOUND;
    }
    
    /* Use 16-bit PCM */
    uint16_t bits_per_sample = 16;
    uint16_t channels = 1;
    uint32_t byte_rate = sample_rate * channels * (bits_per_sample / 8);
    uint16_t block_align = channels * (bits_per_sample / 8);
    uint32_t data_size = (uint32_t)(num_samples * (bits_per_sample / 8));
    uint32_t file_size = 36 + data_size;
    
    /* Write RIFF header */
    WavHeader_RIFF riff;
    memcpy(riff.riff_id, "RIFF", 4);
    riff.file_size = file_size;
    memcpy(riff.wave_id, "WAVE", 4);
    fwrite(&riff, sizeof(riff), 1, f);
    
    /* Write fmt chunk */
    WavHeader_FMT fmt;
    memcpy(fmt.fmt_id, "fmt ", 4);
    fmt.chunk_size = 16;
    fmt.audio_fmt = 1;  /* PCM */
    fmt.num_channels = channels;
    fmt.sample_rate = sample_rate;
    fmt.byte_rate = byte_rate;
    fmt.block_align = block_align;
    fmt.bits_per_sample = bits_per_sample;
    fwrite(&fmt, sizeof(fmt), 1, f);
    
    /* Write data chunk */
    WavHeader_DATA data;
    memcpy(data.data_id, "data", 4);
    data.data_size = data_size;
    fwrite(&data, sizeof(data), 1, f);
    
    /* Write samples */
    for (size_t i = 0; i < num_samples; i++) {
        float s = samples[i];
        
        /* Clamp to [-1, 1] */
        if (s > 1.0f) s = 1.0f;
        if (s < -1.0f) s = -1.0f;
        
        int16_t sample = (int16_t)(s * 32767.0f);
        fwrite(&sample, 2, 1, f);
    }
    
    fclose(f);
    
    return MVE_OK;
}
