# MVE-1: Monologue Voice Engine

## First-Party Single-Speaker Long-Form Synthesis System

### Build Instructions

#### Requirements
- C23-compatible compiler (GCC 13+, Clang 16+)
- POSIX system with pthreads support
- Standard C library (libc, libm)

#### Building

```bash
# Debug build
make debug

# Release build (optimized)
make release

# Clean build artifacts
make clean
```

### Project Structure

```
mve/
├── include/
│   └── mve.h              # Public API header
├── src/
│   ├── main.c             # CLI entry point
│   ├── text_frontend.c    # Text normalization & tokenization
│   ├── wav_io.c           # WAV file I/O
│   ├── timeline.c         # Performance timeline
│   └── utils.c            # Utility functions
├── tests/                 # Test suite
├── tools/                 # Development tools
├── data/
│   └── corpus/            # Training corpus
│       ├── raw/           # Original recordings
│       ├── transcript/    # Text transcripts
│       ├── manifest/      # Corpus metadata
│       ├── derived/       # Processed data
│       └── split/         # Train/val/test splits
└── docs/                  # Documentation
```

### Architecture Overview

MVE-1 implements the following pipeline:

```
TEXT → LINGUISTIC REPRESENTATION → PERFORMANCE PLAN → ACOUSTIC FRAMES → WAVEFORM
```

Key components:

1. **Text Front End**: UTF-8 validation, normalization, tokenization, phoneme conversion
2. **Pronunciation Dictionary**: Custom word-to-phoneme mappings with author overrides
3. **Performance Planner**: Macro and micro prosody planning with authored controls
4. **Acoustic Model**: Duration-regulated synthesis with performance conditioning
5. **Vocoder**: Neural waveform generation from acoustic features
6. **Timeline Engine**: Sample-accurate performance event serialization

### CLI Usage

```bash
# Verify training corpus
mve corpus verify data/corpus/

# Preprocess corpus for training
mve preprocess data/corpus/ build/data/

# Train models
mve train acoustic config.mve
mve train vocoder config.mve
mve train planner config.mve

# Render monologue
mve render monologue.txt --model danny.mve --out episode/

# Inspect generated timeline
mve inspect episode/performance.mvt

# Benchmark model performance
mve bench danny.mve

# Verify model integrity
mve verify danny.mve
```

### Output Files

A successful render produces:

```
episode/
├── narration.wav        # Synthesized audio (24kHz, 16-bit PCM)
├── performance.mvt      # Binary timeline of performance events
├── words.txt            # Word-level timing transcript
└── render.meta          # Rendering metadata
```

### Performance State Dimensions

The engine models performance on two scales:

**Micro-performance** (phoneme/word/phrase):
- Phoneme duration
- F0 (fundamental frequency)
- Energy
- Stress
- Voicing
- Pause duration

**Macro-performance** (paragraph/section/monologue):
- Intensity (0.0 - 1.0)
- Tension (0.0 - 1.0)
- Warmth (0.0 - 1.0)
- Intimacy (0.0 - 1.0)
- Resolve (0.0 - 1.0)
- Pace (0.0 - 1.0)
- Energy trajectory (0.0 - 1.0)

### Authored Performance Controls

Manual direction can override automatic planning:

```
@reflective
@intensity 0.20
@pace 0.87

You keep waiting for certainty.

@pause 900ms

@build intensity=0.62 duration=6s

But certainty was never the requirement.

@resolve

You only had to begin.
```

### Audio Configuration

Default synthesis parameters:
- Sample rate: 24,000 Hz
- FFT size: 1024
- Hop size: 256 samples
- Mel channels: 100
- Frame rate: 93.75 Hz

### Quality Gates

The implementation targets:

- **Gate A**: Perfect intelligibility (no skipped/repeated words)
- **Gate B**: Speaker identity recognition in blind tests
- **Gate C**: Natural short-form speech (<30s)
- **Gate D**: Long-form continuity (10+ minutes)
- **Gate E**: Predictable control response
- **Gate F**: Deterministic output (identical renders)
- **Gate G**: Real-time factor < 1.0

### License

Proprietary - All rights reserved

### Version

MVE-1 v0.1.0 - Initial architecture implementation
