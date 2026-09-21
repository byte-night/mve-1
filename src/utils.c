/* ============================================================================
 * MVE-1: Monologue Voice Engine - Utility Functions
 * 
 * Error strings, version info, and common utilities.
 * ============================================================================
 */

#include "mve.h"
#include <stdio.h>

/* Get error string */
const char* mve_error_string(MVE_Error err) {
    switch (err) {
        case MVE_OK: return "Success";
        case MVE_ERR_GENERAL: return "General error";
        case MVE_ERR_INVALID_ARG: return "Invalid argument";
        case MVE_ERR_OUT_OF_MEMORY: return "Out of memory";
        case MVE_ERR_FILE_NOT_FOUND: return "File not found";
        case MVE_ERR_FILE_FORMAT: return "Invalid file format";
        case MVE_ERR_CORRUPT_DATA: return "Corrupt data";
        case MVE_ERR_NOT_IMPLEMENTED: return "Not implemented";
        case MVE_ERR_TRAINING_FAILED: return "Training failed";
        case MVE_ERR_INFERENCE_FAILED: return "Inference failed";
        default: return "Unknown error";
    }
}

/* Get version string */
const char* mve_version(void) {
    return MVE_VERSION_STRING;
}
