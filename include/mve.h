#ifndef MVE_H
#define MVE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * MVE-1: Monologue Voice Engine v1.0
 * First-Party Single-Speaker Long-Form Synthesis System
 * ============================================================================
 */

/* Version */
#define MVE_VERSION_MAJOR 0
#define MVE_VERSION_MINOR 1
#define MVE_VERSION_PATCH 0
#define MVE_VERSION_STRING "0.1.0"

/* Audio Configuration */
#define MVE_SAMPLE_RATE       24000
#define MVE_FFT_SIZE          1024
#define MVE_HOP_SIZE          256
#define MVE_MEL_CHANNELS      100
#define MVE_FRAME_RATE        (MVE_SAMPLE_RATE / MVE_HOP_SIZE)

/* Performance State Dimensions */
#define MVE_PERF_DIMENSIONS   7

/* Maximum lengths */
#define MVE_MAX_PHONEME_LEN   64
#define MVE_MAX_WORD_LEN      128
#define MVE_MAX_PATH_LEN      4096

/* ============================================================================
 * Error Codes
 * ============================================================================
 */
typedef enum {
    MVE_OK = 0,
    MVE_ERR_GENERAL = -1,
    MVE_ERR_INVALID_ARG = -2,
    MVE_ERR_OUT_OF_MEMORY = -3,
    MVE_ERR_FILE_NOT_FOUND = -4,
    MVE_ERR_FILE_FORMAT = -5,
    MVE_ERR_CORRUPT_DATA = -6,
    MVE_ERR_NOT_IMPLEMENTED = -7,
    MVE_ERR_TRAINING_FAILED = -8,
    MVE_ERR_INFERENCE_FAILED = -9
} MVE_Error;

/* ============================================================================
 * Core Types
 * ============================================================================
 */

/* Phoneme representation */
typedef struct {
    char symbol[MVE_MAX_PHONEME_LEN];
    float duration_frames;
    float f0_hz;
    float energy;
    float stress;
    uint32_t flags;
} MVE_Phoneme;

/* Word representation */
typedef struct {
    char text[MVE_MAX_WORD_LEN];
    char phonemes[10][MVE_MAX_PHONEME_LEN];
    int num_phonemes;
    uint64_t sample_begin;
    uint64_t sample_end;
    float stress;
} MVE_Word;

/* Performance state at time t */
typedef struct {
    float intensity;    /* 0.0 - 1.0 */
    float tension;      /* 0.0 - 1.0 */
    float warmth;       /* 0.0 - 1.0 */
    float intimacy;     /* 0.0 - 1.0 */
    float resolve;      /* 0.0 - 1.0 */
    float pace;         /* 0.0 - 1.0 */
    float energy;       /* 0.0 - 1.0 */
} MVE_PerformanceState;

/* Timeline event types */
typedef enum {
    MVE_EVENT_PHONEME = 0,
    MVE_EVENT_WORD,
    MVE_EVENT_PHRASE,
    MVE_EVENT_SENTENCE,
    MVE_EVENT_PARAGRAPH,
    MVE_EVENT_SECTION,
    MVE_EVENT_PAUSE_BEGIN,
    MVE_EVENT_PAUSE_END,
    MVE_EVENT_BREATH,
    MVE_EVENT_EMPHASIS_BEGIN,
    MVE_EVENT_EMPHASIS_END,
    MVE_EVENT_STATE_KNOT,
    MVE_EVENT_CONTROL_OVERRIDE
} MVE_EventType;

/* Timeline event */
typedef struct {
    uint64_t sample_begin;
    uint64_t sample_end;
    uint32_t type;
    uint32_t item_id;
    float f0_hz;
    float energy;
    float stress;
    float pace;
    float intensity;
    float tension;
    float warmth;
    float resolve;
    uint32_t flags;
} MVE_Event;

/* Acoustic frame */
typedef struct {
    float mel_coeffs[MVE_MEL_CHANNELS];
    float f0_hz;
    float voicing_prob;
    float energy;
    float perf_state[MVE_PERF_DIMENSIONS];
} MVE_AcousticFrame;

/* ============================================================================
 * Text Front End
 * ============================================================================
 */

/* Normalization types */
typedef enum {
    MVE_NORM_PLAIN = 0,
    MVE_NORM_CURRENCY,
    MVE_NORM_DATE,
    MVE_NORM_TIME,
    MVE_NORM_PERCENT,
    MVE_NORM_ABBREV,
    MVE_NORM_ACRONYM
} MVE_NormalizationType;

/* Token types */
typedef enum {
    MVE_TOKEN_WORD = 0,
    MVE_TOKEN_PUNCT,
    MVE_TOKEN_NUMBER,
    MVE_TOKEN_SYMBOL,
    MVE_TOKEN_BREAK
} MVE_TokenType;

typedef struct {
    char text[256];
    MVE_TokenType type;
    MVE_NormalizationType norm_type;
    char spoken_form[256];  /* normalized spoken representation */
} MVE_Token;

/* Forward declarations - opaque types */
typedef struct MVE_TextFrontEnd MVE_TextFrontEnd;
typedef struct MVE_PronunciationDict MVE_PronunciationDict;
typedef struct MVE_AcousticModel MVE_AcousticModel;
typedef struct MVE_Vocoder MVE_Vocoder;
typedef struct MVE_Planner MVE_Planner;
typedef struct MVE_Model MVE_Model;
typedef struct MVE_Corpus MVE_Corpus;
typedef struct MVE_Timeline MVE_Timeline;

/* Timeline structure (needed for CLI inspection) */
struct MVE_Timeline {
    MVE_Event* events;
    size_t num_events;
    size_t capacity;
};

/* ============================================================================
 * Text Front End API
 * ============================================================================
 */

MVE_TextFrontEnd* mve_tfe_create(void);
void mve_tfe_destroy(MVE_TextFrontEnd* tfe);

MVE_Error mve_tfe_normalize(MVE_TextFrontEnd* tfe, const char* input, char* output, size_t output_size);
MVE_Error mve_tfe_tokenize(MVE_TextFrontEnd* tfe, const char* text, MVE_Token* tokens, size_t* num_tokens, size_t max_tokens);
MVE_Error mve_tfe_to_phonemes(MVE_TextFrontEnd* tfe, const MVE_Token* tokens, size_t num_tokens, MVE_Phoneme* phonemes, size_t* num_phonemes, size_t max_phonemes);

/* ============================================================================
 * Pronunciation Dictionary API
 * ============================================================================
 */

MVE_PronunciationDict* mve_prondict_create(void);
void mve_prondict_destroy(MVE_PronunciationDict* dict);

MVE_Error mve_prondict_load(MVE_PronunciationDict* dict, const char* filepath);
MVE_Error mve_prondict_add(MVE_PronunciationDict* dict, const char* word, const char* phonemes);
MVE_Error mve_prondict_lookup(MVE_PronunciationDict* dict, const char* word, char* phonemes, size_t max_len);

/* ============================================================================
 * Model Loading
 * ============================================================================
 */

MVE_Model* mve_model_create(void);
void mve_model_destroy(MVE_Model* model);

MVE_Error mve_model_load(MVE_Model* model, const char* filepath);
MVE_Error mve_model_save(MVE_Model* model, const char* filepath);

/* ============================================================================
 * Acoustic Model API
 * ============================================================================
 */

MVE_AcousticModel* mve_acoustic_create(const MVE_Model* model);
void mve_acoustic_destroy(MVE_AcousticModel* am);

MVE_Error mve_acoustic_forward(
    MVE_AcousticModel* am,
    const MVE_Phoneme* phonemes,
    size_t num_phonemes,
    const MVE_PerformanceState* perf_state,
    MVE_AcousticFrame* frames,
    size_t* num_frames,
    size_t max_frames
);

/* ============================================================================
 * Vocoder API
 * ============================================================================
 */

MVE_Vocoder* mve_vocoder_create(const MVE_Model* model);
void mve_vocoder_destroy(MVE_Vocoder* voc);

MVE_Error mve_vocoder_synthesize(
    MVE_Vocoder* voc,
    const MVE_AcousticFrame* frames,
    size_t num_frames,
    float* waveform,
    size_t* num_samples,
    size_t max_samples
);

/* ============================================================================
 * Performance Planner API
 * ============================================================================
 */

MVE_Planner* mve_planner_create(const MVE_Model* model);
void mve_planner_destroy(MVE_Planner* planner);

MVE_Error mve_planner_plan(
    MVE_Planner* planner,
    const MVE_Token* tokens,
    size_t num_tokens,
    MVE_PerformanceState* states,
    size_t num_states
);

MVE_Error mve_planner_apply_control(
    MVE_Planner* planner,
    const char* control_str,
    MVE_PerformanceState* state
);

/* ============================================================================
 * Timeline API
 * ============================================================================
 */

MVE_Timeline* mve_timeline_create(void);
void mve_timeline_destroy(MVE_Timeline* timeline);

MVE_Error mve_timeline_add_event(MVE_Timeline* timeline, const MVE_Event* event);
MVE_Error mve_timeline_write(MVE_Timeline* timeline, const char* filepath);
MVE_Error mve_timeline_read(MVE_Timeline* timeline, const char* filepath);

/* ============================================================================
 * Corpus API
 * ============================================================================
 */

MVE_Corpus* mve_corpus_create(void);
void mve_corpus_destroy(MVE_Corpus* corpus);

MVE_Error mve_corpus_scan(MVE_Corpus* corpus, const char* root_path);
MVE_Error mve_corpus_verify(MVE_Corpus* corpus);
MVE_Error mve_corpus_preprocess(MVE_Corpus* corpus, const char* output_path);

/* ============================================================================
 * Training API
 * ============================================================================
 */

typedef enum {
    MVE_TRAIN_ACOUSTIC = 0,
    MVE_TRAIN_VOCODER,
    MVE_TRAIN_PLANNER,
    MVE_TRAIN_ALIGNMENT
} MVE_TrainPhase;

typedef struct {
    MVE_TrainPhase phase;
    const char* config_path;
    const char* output_dir;
    int epochs;
    int batch_size;
    float learning_rate;
    int save_interval;
} MVE_TrainConfig;

MVE_Error mve_train(const MVE_TrainConfig* config);

/* ============================================================================
 * Rendering API
 * ============================================================================
 */

typedef struct {
    const char* text_path;
    const char* model_path;
    const char* output_dir;
    int sample_rate;
    int deterministic;
} MVE_RenderConfig;

typedef struct {
    char wav_path[MVE_MAX_PATH_LEN];
    char mvt_path[MVE_MAX_PATH_LEN];
    char words_path[MVE_MAX_PATH_LEN];
    char meta_path[MVE_MAX_PATH_LEN];
    uint64_t total_samples;
    double duration_seconds;
} MVE_RenderResult;

MVE_Error mve_render(const MVE_RenderConfig* config, MVE_RenderResult* result);

/* ============================================================================
 * Utility API
 * ============================================================================
 */

const char* mve_error_string(MVE_Error err);
const char* mve_version(void);

/* DSP utilities */
MVE_Error mve_wav_read(const char* path, float** samples, size_t* num_samples, int* sample_rate);
MVE_Error mve_wav_write(const char* path, const float* samples, size_t num_samples, int sample_rate);

/* Benchmarking */
typedef struct {
    double inference_time_ms;
    double rt_factor;
    size_t peak_memory_bytes;
} MVE_BenchResult;

MVE_Error mve_bench(const char* model_path, MVE_BenchResult* result);

#ifdef __cplusplus
}
#endif

#endif /* MVE_H */
