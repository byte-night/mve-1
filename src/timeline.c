/* ============================================================================
 * MVE-1: Monologue Voice Engine - Timeline Implementation
 * 
 * Performance timeline (performance.mvt) generation and serialization.
 * Sample-accurate timing for all performance events.
 * ============================================================================
 */

#include "mve.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Internal structure - defined in header, implemented here */

/* Create timeline */
MVE_Timeline* mve_timeline_create(void) {
    MVE_Timeline* tl = (MVE_Timeline*)calloc(1, sizeof(MVE_Timeline));
    if (!tl) return NULL;
    
    tl->capacity = 4096;
    tl->events = (MVE_Event*)calloc(tl->capacity, sizeof(MVE_Event));
    if (!tl->events) {
        free(tl);
        return NULL;
    }
    
    return tl;
}

/* Destroy timeline */
void mve_timeline_destroy(MVE_Timeline* tl) {
    if (!tl) return;
    if (tl->events) free(tl->events);
    free(tl);
}

/* Add event to timeline */
MVE_Error mve_timeline_add_event(MVE_Timeline* tl, const MVE_Event* event) {
    if (!tl || !event) {
        return MVE_ERR_INVALID_ARG;
    }
    
    /* Grow if needed */
    if (tl->num_events >= tl->capacity) {
        size_t new_cap = tl->capacity * 2;
        MVE_Event* new_events = (MVE_Event*)realloc(tl->events, new_cap * sizeof(MVE_Event));
        if (!new_events) {
            return MVE_ERR_OUT_OF_MEMORY;
        }
        tl->events = new_events;
        tl->capacity = new_cap;
    }
    
    tl->events[tl->num_events++] = *event;
    return MVE_OK;
}

/* Write timeline to binary file */
MVE_Error mve_timeline_write(MVE_Timeline* tl, const char* filepath) {
    if (!tl || !filepath) {
        return MVE_ERR_INVALID_ARG;
    }
    
    FILE* f = fopen(filepath, "wb");
    if (!f) {
        return MVE_ERR_FILE_NOT_FOUND;
    }
    
    /* Write magic header "MVT1" */
    const char magic[4] = {'M', 'V', 'T', '1'};
    fwrite(magic, 4, 1, f);
    
    /* Write version */
    uint32_t version = 1;
    fwrite(&version, 4, 1, f);
    
    /* Write event count */
    uint32_t count = (uint32_t)tl->num_events;
    fwrite(&count, 4, 1, f);
    
    /* Write events with explicit field order (no padding) */
    for (size_t i = 0; i < tl->num_events; i++) {
        const MVE_Event* ev = &tl->events[i];
        
        /* Write fields in defined order, little-endian */
        fwrite(&ev->sample_begin, 8, 1, f);
        fwrite(&ev->sample_end, 8, 1, f);
        fwrite(&ev->type, 4, 1, f);
        fwrite(&ev->item_id, 4, 1, f);
        fwrite(&ev->f0_hz, 4, 1, f);
        fwrite(&ev->energy, 4, 1, f);
        fwrite(&ev->stress, 4, 1, f);
        fwrite(&ev->pace, 4, 1, f);
        fwrite(&ev->intensity, 4, 1, f);
        fwrite(&ev->tension, 4, 1, f);
        fwrite(&ev->warmth, 4, 1, f);
        fwrite(&ev->resolve, 4, 1, f);
        fwrite(&ev->flags, 4, 1, f);
    }
    
    fclose(f);
    return MVE_OK;
}

/* Read timeline from binary file */
MVE_Error mve_timeline_read(MVE_Timeline* tl, const char* filepath) {
    if (!tl || !filepath) {
        return MVE_ERR_INVALID_ARG;
    }
    
    FILE* f = fopen(filepath, "rb");
    if (!f) {
        return MVE_ERR_FILE_NOT_FOUND;
    }
    
    /* Read and verify magic */
    char magic[4];
    if (fread(magic, 4, 1, f) != 1 || memcmp(magic, "MVT1", 4) != 0) {
        fclose(f);
        return MVE_ERR_FILE_FORMAT;
    }
    
    /* Read version */
    uint32_t version;
    if (fread(&version, 4, 1, f) != 1 || version != 1) {
        fclose(f);
        return MVE_ERR_FILE_FORMAT;
    }
    
    /* Read event count */
    uint32_t count;
    if (fread(&count, 4, 1, f) != 1) {
        fclose(f);
        return MVE_ERR_FILE_FORMAT;
    }
    
    /* Allocate space */
    if (count > tl->capacity) {
        MVE_Event* new_events = (MVE_Event*)realloc(tl->events, count * sizeof(MVE_Event));
        if (!new_events) {
            fclose(f);
            return MVE_ERR_OUT_OF_MEMORY;
        }
        tl->events = new_events;
        tl->capacity = count;
    }
    
    /* Read events */
    for (uint32_t i = 0; i < count; i++) {
        MVE_Event* ev = &tl->events[i];
        
        fread(&ev->sample_begin, 8, 1, f);
        fread(&ev->sample_end, 8, 1, f);
        fread(&ev->type, 4, 1, f);
        fread(&ev->item_id, 4, 1, f);
        fread(&ev->f0_hz, 4, 1, f);
        fread(&ev->energy, 4, 1, f);
        fread(&ev->stress, 4, 1, f);
        fread(&ev->pace, 4, 1, f);
        fread(&ev->intensity, 4, 1, f);
        fread(&ev->tension, 4, 1, f);
        fread(&ev->warmth, 4, 1, f);
        fread(&ev->resolve, 4, 1, f);
        fread(&ev->flags, 4, 1, f);
    }
    
    tl->num_events = count;
    
    fclose(f);
    return MVE_OK;
}
