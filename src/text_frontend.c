/* ============================================================================
 * MVE-1: Monologue Voice Engine - Text Front End Implementation
 * 
 * Handles:
 * - UTF-8 validation
 * - Text normalization (numbers, currency, dates, times, abbreviations)
 * - Sentence segmentation
 * - Word tokenization
 * - Pronunciation lookup
 * - Phoneme conversion
 * ============================================================================
 */

#include "mve.h"
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdio.h>

/* Internal structure */
struct MVE_TextFrontEnd {
    int initialized;
    char* norm_buffer;
    size_t norm_buffer_size;
};

/* Create text front end */
MVE_TextFrontEnd* mve_tfe_create(void) {
    MVE_TextFrontEnd* tfe = (MVE_TextFrontEnd*)calloc(1, sizeof(MVE_TextFrontEnd));
    if (!tfe) return NULL;
    
    tfe->norm_buffer_size = 65536;
    tfe->norm_buffer = (char*)malloc(tfe->norm_buffer_size);
    if (!tfe->norm_buffer) {
        free(tfe);
        return NULL;
    }
    
    tfe->initialized = 1;
    return tfe;
}

/* Destroy text front end */
void mve_tfe_destroy(MVE_TextFrontEnd* tfe) {
    if (!tfe) return;
    if (tfe->norm_buffer) free(tfe->norm_buffer);
    free(tfe);
}

/* Validate UTF-8 input - returns 1 if valid, 0 if invalid */
static int validate_utf8(const char* str, size_t len) {
    if (!str) return 0;
    
    size_t i = 0;
    while (i < len) {
        unsigned char c = (unsigned char)str[i];
        
        if (c == 0) break;
        
        if (c < 0x80) {
            /* ASCII */
            i++;
        } else if ((c & 0xE0) == 0xC0) {
            /* 2-byte sequence */
            if (i + 1 >= len || (str[i+1] & 0xC0) != 0x80) return 0;
            i += 2;
        } else if ((c & 0xF0) == 0xE0) {
            /* 3-byte sequence */
            if (i + 2 >= len || (str[i+1] & 0xC0) != 0x80 || (str[i+2] & 0xC0) != 0x80) return 0;
            i += 3;
        } else if ((c & 0xF8) == 0xF0) {
            /* 4-byte sequence */
            if (i + 3 >= len || (str[i+1] & 0xC0) != 0x80 || (str[i+2] & 0xC0) != 0x80 || (str[i+3] & 0xC0) != 0x80) return 0;
            i += 4;
        } else {
            return 0;
        }
    }
    return 1;
}

/* Normalize a number like "3.7" to "three point seven" */
static void normalize_number(const char* num_str, char* out, size_t out_size) {
    (void)num_str;
    (void)out;
    (void)out_size;
    /* TODO: Implement full number-to-words conversion */
}

/* Normalize currency like "$100" to "one hundred dollars" */
static void normalize_currency(const char* curr_str, char* out, size_t out_size) {
    (void)curr_str;
    (void)out;
    (void)out_size;
    /* TODO: Implement full currency normalization */
}

/* Check if character is punctuation that indicates a break */
static int is_sentence_break(char c) {
    return c == '.' || c == '!' || c == '?' || c == '\n';
}

static int is_phrase_break(char c) {
    return c == ',' || c == ';' || c == ':' || c == '-' || c == '"';
}

/* Normalize text - expand numbers, currency, abbreviations, etc. */
MVE_Error mve_tfe_normalize(MVE_TextFrontEnd* tfe, const char* input, char* output, size_t output_size) {
    if (!tfe || !input || !output || output_size == 0) {
        return MVE_ERR_INVALID_ARG;
    }
    
    if (!validate_utf8(input, strlen(input))) {
        return MVE_ERR_FILE_FORMAT;
    }
    
    size_t in_len = strlen(input);
    size_t out_idx = 0;
    size_t i = 0;
    
    while (i < in_len && out_idx < output_size - 1) {
        /* Check for numbers with decimals like "3.7" */
        if (isdigit((unsigned char)input[i])) {
            char num_buf[64] = {0};
            size_t num_idx = 0;
            
            /* Collect digits and decimal point */
            while (i < in_len && (isdigit((unsigned char)input[i]) || input[i] == '.') && num_idx < 63) {
                num_buf[num_idx++] = input[i++];
            }
            
            /* Simple passthrough - full implementation would convert to words */
            for (size_t j = 0; j < num_idx && out_idx < output_size - 1; j++) {
                output[out_idx++] = num_buf[j];
            }
            continue;
        }
        
        /* Check for currency symbols - use byte comparison for UTF-8 */
        if (input[i] == '$') {
            output[out_idx++] = input[i++];
            continue;
        }
        /* Skip € and £ handling for now - simplified ASCII-only normalization */
        
        /* Regular character */
        output[out_idx++] = input[i++];
    }
    
    output[out_idx] = '\0';
    return MVE_OK;
}

/* Tokenize normalized text into words, punctuation, numbers, etc. */
MVE_Error mve_tfe_tokenize(MVE_TextFrontEnd* tfe, const char* text, MVE_Token* tokens, size_t* num_tokens, size_t max_tokens) {
    if (!tfe || !text || !tokens || !num_tokens || max_tokens == 0) {
        return MVE_ERR_INVALID_ARG;
    }
    
    size_t token_idx = 0;
    size_t i = 0;
    size_t text_len = strlen(text);
    
    while (i < text_len && token_idx < max_tokens) {
        /* Skip whitespace */
        while (i < text_len && isspace((unsigned char)text[i])) {
            i++;
        }
        
        if (i >= text_len) break;
        
        MVE_Token* tok = &tokens[token_idx];
        memset(tok, 0, sizeof(MVE_Token));
        
        /* Check for sentence/phrase breaks */
        if (is_sentence_break(text[i]) || is_phrase_break(text[i])) {
            tok->type = MVE_TOKEN_PUNCT;
            tok->text[0] = text[i];
            tok->text[1] = '\0';
            strcpy(tok->spoken_form, tok->text);
            tok->norm_type = MVE_NORM_PLAIN;
            i++;
            token_idx++;
            continue;
        }
        
        /* Collect word/number */
        size_t word_start = i;
        while (i < text_len && !isspace((unsigned char)text[i]) && !is_sentence_break(text[i]) && !is_phrase_break(text[i])) {
            i++;
        }
        
        size_t word_len = i - word_start;
        if (word_len > 255) word_len = 255;
        
        strncpy(tok->text, &text[word_start], word_len);
        tok->text[word_len] = '\0';
        
        /* Determine token type */
        int all_digits = 1;
        int has_digit = 0;
        for (size_t j = 0; j < word_len; j++) {
            char c = tok->text[j];
            if (isdigit((unsigned char)c)) {
                has_digit = 1;
            } else if (!isalpha((unsigned char)c) && c != '.' && c != '-' && c != '$') {
                all_digits = 0;
            }
        }
        
        if (has_digit && all_digits) {
            tok->type = MVE_TOKEN_NUMBER;
            tok->norm_type = MVE_NORM_PLAIN;
        } else if (has_digit) {
            tok->type = MVE_TOKEN_NUMBER;
            tok->norm_type = MVE_NORM_PLAIN;
        } else {
            tok->type = MVE_TOKEN_WORD;
            tok->norm_type = MVE_NORM_PLAIN;
        }
        
        /* Spoken form initially same as text */
        strncpy(tok->spoken_form, tok->text, 255);
        tok->spoken_form[255] = '\0';
        
        token_idx++;
    }
    
    *num_tokens = token_idx;
    return MVE_OK;
}

/* Convert tokens to phoneme sequence */
MVE_Error mve_tfe_to_phonemes(MVE_TextFrontEnd* tfe, const MVE_Token* tokens, size_t num_tokens, MVE_Phoneme* phonemes, size_t* num_phonemes, size_t max_phonemes) {
    if (!tfe || !tokens || !phonemes || !num_phonemes || max_phonemes == 0) {
        return MVE_ERR_INVALID_ARG;
    }
    
    size_t phoneme_idx = 0;
    
    for (size_t i = 0; i < num_tokens && phoneme_idx < max_phonemes; i++) {
        const MVE_Token* tok = &tokens[i];
        
        if (tok->type == MVE_TOKEN_BREAK || tok->type == MVE_TOKEN_PUNCT) {
            /* Add pause marker */
            MVE_Phoneme* ph = &phonemes[phoneme_idx++];
            memset(ph, 0, sizeof(MVE_Phoneme));
            strcpy(ph->symbol, "<pause>");
            ph->duration_frames = 10.0f;  /* Default pause duration */
            continue;
        }
        
        /* For now, use grapheme-to-phoneme placeholder */
        /* Full implementation would use pronunciation dictionary or G2P model */
        const char* text = tok->spoken_form;
        size_t len = strlen(text);
        
        for (size_t j = 0; j < len && phoneme_idx < max_phonemes; j++) {
            char c = text[j];
            
            /* Skip non-alphabetic */
            if (!isalpha((unsigned char)c)) continue;
            
            MVE_Phoneme* ph = &phonemes[phoneme_idx++];
            memset(ph, 0, sizeof(MVE_Phoneme));
            
            /* Very simplified - each letter becomes a phoneme */
            /* Real implementation uses proper phoneme inventory */
            ph->symbol[0] = tolower((unsigned char)c);
            ph->symbol[1] = '\0';
            ph->duration_frames = 3.0f;  /* Default duration */
            ph->f0_hz = 150.0f;  /* Default F0 */
            ph->energy = 0.5f;
        }
    }
    
    *num_phonemes = phoneme_idx;
    return MVE_OK;
}
