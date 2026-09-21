/* ============================================================================
 * MVE-1: Monologue Voice Engine - Main CLI Entry Point
 * 
 * Command-line interface for all MVE operations:
 *   mve corpus verify <path>
 *   mve preprocess <corpus> <output>
 *   mve train <phase> <config>
 *   mve render <text> --model <model> --out <dir>
 *   mve inspect <timeline>
 *   mve bench <model>
 *   mve verify <model>
 * ============================================================================
 */

#include "mve.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_usage(void) {
    printf("MVE-1 Monologue Voice Engine v%s\n", mve_version());
    printf("\n");
    printf("Usage:\n");
    printf("  mve corpus verify <corpus_path>\n");
    printf("  mve preprocess <corpus_path> <output_path>\n");
    printf("  mve train <phase> <config_path>\n");
    printf("  mve render <text_path> --model <model_path> --out <output_dir>\n");
    printf("  mve inspect <timeline_path>\n");
    printf("  mve bench <model_path>\n");
    printf("  mve verify <model_path>\n");
    printf("\n");
    printf("Training phases:\n");
    printf("  acoustic    Train acoustic model\n");
    printf("  vocoder     Train neural vocoder\n");
    printf("  planner     Train performance planner\n");
    printf("  alignment   Train alignment model\n");
    printf("\n");
}

static int cmd_corpus_verify(const char* corpus_path) {
    printf("Verifying corpus: %s\n", corpus_path);
    
    MVE_Corpus* corpus = mve_corpus_create();
    if (!corpus) {
        fprintf(stderr, "Failed to create corpus handle\n");
        return 1;
    }
    
    MVE_Error err = mve_corpus_scan(corpus, corpus_path);
    if (err != MVE_OK) {
        fprintf(stderr, "Failed to scan corpus: %s\n", mve_error_string(err));
        mve_corpus_destroy(corpus);
        return 1;
    }
    
    err = mve_corpus_verify(corpus);
    if (err != MVE_OK) {
        fprintf(stderr, "Corpus verification failed: %s\n", mve_error_string(err));
        mve_corpus_destroy(corpus);
        return 1;
    }
    
    printf("Corpus verification successful\n");
    mve_corpus_destroy(corpus);
    return 0;
}

static int cmd_render(const char* text_path, const char* model_path, const char* output_dir) {
    printf("Rendering monologue...\n");
    printf("  Text: %s\n", text_path);
    printf("  Model: %s\n", model_path);
    printf("  Output: %s\n", output_dir);
    
    MVE_RenderConfig config = {
        .text_path = text_path,
        .model_path = model_path,
        .output_dir = output_dir,
        .sample_rate = MVE_SAMPLE_RATE,
        .deterministic = 1
    };
    
    MVE_RenderResult result;
    memset(&result, 0, sizeof(result));
    
    MVE_Error err = mve_render(&config, &result);
    if (err != MVE_OK) {
        fprintf(stderr, "Render failed: %s\n", mve_error_string(err));
        return 1;
    }
    
    printf("Render complete:\n");
    printf("  WAV: %s\n", result.wav_path);
    printf("  Timeline: %s\n", result.mvt_path);
    printf("  Duration: %.2f seconds\n", result.duration_seconds);
    printf("  Samples: %lu\n", (unsigned long)result.total_samples);
    
    return 0;
}

static int cmd_inspect(const char* timeline_path) {
    printf("Inspecting timeline: %s\n", timeline_path);
    
    MVE_Timeline* tl = mve_timeline_create();
    if (!tl) {
        fprintf(stderr, "Failed to create timeline handle\n");
        return 1;
    }
    
    MVE_Error err = mve_timeline_read(tl, timeline_path);
    if (err != MVE_OK) {
        fprintf(stderr, "Failed to read timeline: %s\n", mve_error_string(err));
        mve_timeline_destroy(tl);
        return 1;
    }
    
    printf("Timeline contains %zu events\n", tl->num_events);
    
    /* Print summary by event type */
    int type_counts[14] = {0};
    uint64_t total_samples = 0;
    
    for (size_t i = 0; i < tl->num_events; i++) {
        MVE_Event* ev = &tl->events[i];
        if (ev->type < 14) {
            type_counts[ev->type]++;
        }
        if (ev->sample_end > total_samples) {
            total_samples = ev->sample_end;
        }
    }
    
    const char* type_names[] = {
        "PHONEME", "WORD", "PHRASE", "SENTENCE", "PARAGRAPH", "SECTION",
        "PAUSE_BEGIN", "PAUSE_END", "BREATH", "EMPHASIS_BEGIN", "EMPHASIS_END",
        "STATE_KNOT", "CONTROL_OVERRIDE"
    };
    
    printf("\nEvent summary:\n");
    for (int i = 0; i < 13; i++) {
        if (type_counts[i] > 0) {
            printf("  %-16s %d\n", type_names[i], type_counts[i]);
        }
    }
    
    double duration = (double)total_samples / MVE_SAMPLE_RATE;
    printf("\nTotal duration: %.2f seconds (%.2f minutes)\n", duration, duration / 60.0);
    
    mve_timeline_destroy(tl);
    return 0;
}

static int cmd_bench(const char* model_path) {
    printf("Benchmarking model: %s\n", model_path);
    
    MVE_BenchResult result;
    memset(&result, 0, sizeof(result));
    
    MVE_Error err = mve_bench(model_path, &result);
    if (err != MVE_OK) {
        fprintf(stderr, "Benchmark failed: %s\n", mve_error_string(err));
        return 1;
    }
    
    printf("Benchmark results:\n");
    printf("  Inference time: %.2f ms\n", result.inference_time_ms);
    printf("  Real-time factor: %.3f\n", result.rt_factor);
    printf("  Peak memory: %.2f MB\n", result.peak_memory_bytes / (1024.0 * 1024.0));
    
    return 0;
}

static int cmd_verify(const char* model_path) {
    printf("Verifying model: %s\n", model_path);
    
    /* Load and validate model structure */
    MVE_Model* model = mve_model_create();
    if (!model) {
        fprintf(stderr, "Failed to create model handle\n");
        return 1;
    }
    
    MVE_Error err = mve_model_load(model, model_path);
    if (err != MVE_OK) {
        fprintf(stderr, "Model verification failed: %s\n", mve_error_string(err));
        mve_model_destroy(model);
        return 1;
    }
    
    printf("Model verification successful\n");
    mve_model_destroy(model);
    return 0;
}

static int cmd_train(const char* phase_str, const char* config_path) {
    MVE_TrainPhase phase;
    
    if (strcmp(phase_str, "acoustic") == 0) {
        phase = MVE_TRAIN_ACOUSTIC;
    } else if (strcmp(phase_str, "vocoder") == 0) {
        phase = MVE_TRAIN_VOCODER;
    } else if (strcmp(phase_str, "planner") == 0) {
        phase = MVE_TRAIN_PLANNER;
    } else if (strcmp(phase_str, "alignment") == 0) {
        phase = MVE_TRAIN_ALIGNMENT;
    } else {
        fprintf(stderr, "Unknown training phase: %s\n", phase_str);
        return 1;
    }
    
    printf("Starting training phase: %s\n", phase_str);
    printf("Config: %s\n", config_path);
    
    MVE_TrainConfig config = {
        .phase = phase,
        .config_path = config_path,
        .output_dir = "./output",
        .epochs = 100,
        .batch_size = 32,
        .learning_rate = 0.001f,
        .save_interval = 10
    };
    
    MVE_Error err = mve_train(&config);
    if (err != MVE_OK) {
        fprintf(stderr, "Training failed: %s\n", mve_error_string(err));
        return 1;
    }
    
    printf("Training complete\n");
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return 1;
    }
    
    const char* cmd = argv[1];
    
    if (strcmp(cmd, "--help") == 0 || strcmp(cmd, "-h") == 0) {
        print_usage();
        return 0;
    }
    
    if (strcmp(cmd, "--version") == 0 || strcmp(cmd, "-v") == 0) {
        printf("MVE-1 v%s\n", mve_version());
        return 0;
    }
    
    if (strcmp(cmd, "corpus") == 0) {
        if (argc < 4) {
            fprintf(stderr, "Usage: mve corpus verify <path>\n");
            return 1;
        }
        return cmd_corpus_verify(argv[3]);
    }
    
    if (strcmp(cmd, "preprocess") == 0) {
        if (argc < 4) {
            fprintf(stderr, "Usage: mve preprocess <corpus> <output>\n");
            return 1;
        }
        printf("Preprocessing not yet implemented\n");
        return 1;
    }
    
    if (strcmp(cmd, "train") == 0) {
        if (argc < 4) {
            fprintf(stderr, "Usage: mve train <phase> <config>\n");
            return 1;
        }
        return cmd_train(argv[2], argv[3]);
    }
    
    if (strcmp(cmd, "render") == 0) {
        if (argc < 3) {
            fprintf(stderr, "Usage: mve render <text> --model <model> --out <dir>\n");
            return 1;
        }
        
        const char* text_path = argv[2];
        const char* model_path = NULL;
        const char* output_dir = "./output";
        
        for (int i = 3; i < argc; i++) {
            if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
                model_path = argv[++i];
            } else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
                output_dir = argv[++i];
            }
        }
        
        if (!model_path) {
            fprintf(stderr, "Error: --model is required\n");
            return 1;
        }
        
        return cmd_render(text_path, model_path, output_dir);
    }
    
    if (strcmp(cmd, "inspect") == 0) {
        if (argc < 3) {
            fprintf(stderr, "Usage: mve inspect <timeline>\n");
            return 1;
        }
        return cmd_inspect(argv[2]);
    }
    
    if (strcmp(cmd, "bench") == 0) {
        if (argc < 3) {
            fprintf(stderr, "Usage: mve bench <model>\n");
            return 1;
        }
        return cmd_bench(argv[2]);
    }
    
    if (strcmp(cmd, "verify") == 0) {
        if (argc < 3) {
            fprintf(stderr, "Usage: mve verify <model>\n");
            return 1;
        }
        return cmd_verify(argv[2]);
    }
    
    fprintf(stderr, "Unknown command: %s\n", cmd);
    print_usage();
    return 1;
}
