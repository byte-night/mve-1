# MVE-1

## First-Party Monologue Voice Engine

### Architecture & Implementation Specification v0.1

**Author:** Danny Dekker
**Status:** Initial architecture specification
**Implementation target:** C23 / first-party runtime
**Primary application:** High-fidelity long-form synthetic performance of the author's own voice

---

## Abstract

MVE-1 is a first-party, proprietary speech synthesis system designed specifically for long-form monologue performance.

The system is not intended to be a general-purpose multi-speaker TTS service. Its purpose is narrower:

> Given an authored monologue, synthesize a continuous vocal performance that is recognizably the target speaker, preserves intelligibility and identity over long durations, follows an intentional emotional arc, exposes precise directorial controls, and emits a machine-readable performance timeline synchronized exactly to the generated waveform.

The complete required execution path is implemented internally in C23.

There is no required Python runtime, PyTorch, TensorFlow, ONNX, external DSP library, external vocoder, external text processor, or cloud inference service.

The system owns:

* text normalization
* pronunciation representation
* audio ingestion
* resampling
* FFT/STFT
* acoustic feature extraction
* alignment
* neural-network kernels
* training
* optimization
* acoustic synthesis
* waveform synthesis
* model serialization
* inference
* performance planning
* performance-timeline generation
* CLI tooling
* verification and benchmarking

The principal design decision is to separate **what is said**, **how it should be performed**, and **how that performance becomes audio**.

MVE-1 therefore treats prosody and performance state as first-class representations rather than accidental side effects of a black-box text-to-waveform network.

---

# 1. Objective

The success criterion is not:

> “Does this sound like Danny for fifteen seconds?”

It is:

> **Can MVE-1 render a ten-minute monologue such that a listener could reasonably believe it was performed deliberately as a continuous studio recording?**

The target includes:

* speaker identity
* intelligibility
* stable pronunciation
* natural pauses
* sentence-to-sentence continuity
* emotional progression
* non-repetitive cadence
* controllable emphasis
* deliberate silence
* sustained long-form coherence

A synthesized paragraph must not sound like ten independently generated sentences concatenated together.

The monologue itself is the unit of performance.

---

# 2. System Principles

## 2.1 First party

The canonical MVE implementation SHALL be owned and implemented internally.

Published research may inform architectural decisions but SHALL NOT create runtime dependencies.

The baseline build SHALL require only:

```text
C23
libc
libm
pthreads
OS file/memory primitives
```

A hardware-accelerated backend may eventually exist, but the CPU implementation remains the reference implementation.

---

## 2.2 Single speaker first

MVE-1 deliberately rejects the complexity of universal voice cloning.

The initial system knows one speaker extremely well.

This allows model capacity to be spent on:

```text
identity
pronunciation
prosody
timing
emotion
long-form continuity
```

rather than generalized speaker adaptation.

---

## 2.3 Performance is explicit

Speech cannot be represented as:

```text
text → waveform
```

alone.

MVE defines:

```text
TEXT
  │
  ▼
LINGUISTIC REPRESENTATION
  │
  ▼
PERFORMANCE PLAN
  │
  ▼
ACOUSTIC PERFORMANCE
  │
  ▼
WAVEFORM
```

Duration, pitch and energy are established controllable variables in modern TTS systems; FastSpeech 2, for example, explicitly conditions synthesis on duration, pitch and energy rather than forcing the decoder to hide all variation internally.

MVE extends that philosophy upward from individual utterances to the entire monologue.

---

# 3. Architectural Overview

```text
                         MONOLOGUE
                             │
                             ▼
                    ┌─────────────────┐
                    │  TEXT FRONT END │
                    └────────┬────────┘
                             │
                    linguistic sequence
                             │
                             ▼
                  ┌─────────────────────┐
                  │ PERFORMANCE PLANNER │
                  └──────────┬──────────┘
                             │
                performance trajectory
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
     ┌─────────────────┐          ┌─────────────────┐
     │ ACOUSTIC ENGINE │          │ TIMELINE ENGINE │
     └────────┬────────┘          └────────┬────────┘
              │                            │
         acoustic frames              semantic /
              │                       performance
              ▼                         events
       ┌──────────────┐                   │
       │   VOCODER    │                   │
       └──────┬───────┘                   │
              │                            │
              ▼                            ▼
       narration.wav              performance.mvt
```

The audio and timeline originate from the same internal performance representation.

The timeline is therefore authoritative rather than reconstructed afterward from audio analysis.

---

# 4. Text Front End

MVE SHALL contain its own UTF-8 text processor.

The front end performs:

```text
UTF-8 validation
        ↓
normalization
        ↓
sentence segmentation
        ↓
phrase segmentation
        ↓
word representation
        ↓
pronunciation
        ↓
phoneme sequence
```

Normalization includes at minimum:

* integers
* decimals
* percentages
* currency
* dates
* times
* abbreviations
* acronyms
* punctuation
* quotations
* symbols
* units

For example:

```text
3.7 GHz
```

may become an internal spoken representation equivalent to:

```text
three point seven gigahertz
```

without changing the original source text.

---

# 5. Pronunciation Layer

The acoustic network SHALL NOT be responsible for guessing arbitrary English pronunciation directly from raw spelling.

MVE uses an internal phoneme inventory.

A word can resolve through:

```text
internal pronunciation dictionary
        ↓
letter-to-sound rules
        ↓
learned grapheme-to-phoneme predictor
        ↓
explicit author override
```

The override mechanism is authoritative.

Example:

```text
@pronounce "RoPE" = /roʊp/
```

or an eventual compact internal notation.

This is particularly important for:

* technical terminology
* proper names
* acronyms
* uncommon words
* deliberately unusual pronunciation

No external pronunciation service is required during rendering.

---

# 6. Performance Representation

This is the central architectural feature of MVE.

The engine SHALL model performance on at least two temporal scales.

## Micro-performance

```text
phoneme
word
phrase
sentence
```

Controls:

```text
phoneme duration
F0
energy
stress
voicing
pause duration
breath
local pace
```

## Macro-performance

```text
paragraph
section
monologue
```

Controls:

```text
intensity
tension
warmth
certainty
intimacy
pace
energy trajectory
pitch range
pause tendency
```

The dimensions are performance controls, not claims about the semantic truth of the text.

---

# 7. Macro Performance State

Define a continuous state:

```text
P(t) =
[
    intensity,
    tension,
    warmth,
    intimacy,
    resolve,
    pace,
    energy
]
```

where each component is normalized.

Example:

```text
opening

intensity = 0.18
tension   = 0.12
warmth    = 0.32
resolve   = 0.20

        ↓

argument develops

intensity = 0.43
tension   = 0.51
resolve   = 0.38

        ↓

climax

intensity = 0.88
tension   = 0.71
resolve   = 0.84

        ↓

resolution

intensity = 0.31
tension   = 0.08
resolve   = 0.93
```

Research on paragraph-level synthesis has found that cross-sentence linguistic and prosodic context can improve long-form speech compared with treating sentences independently.

MVE therefore maintains persistent performance state across sentence boundaries.

---

# 8. Performance Planner

The planner converts a document into an explicit performance score.

```text
document
   ↓
paragraph encoder
   ↓
sentence states
   ↓
phrase states
   ↓
phoneme targets
```

The initial learned planner SHOULD use a small recurrent architecture rather than requiring a large language model inside the synthesis runtime.

A persistent state vector is updated as the system traverses the monologue.

Conceptually:

```text
S[n+1] = F(
    S[n],
    sentence[n],
    document_position,
    authored_controls
)
```

This gives the next sentence knowledge of:

* what came before
* how intense the performance currently is
* where the piece is headed
* its location within the monologue
* any directorial instructions

A small recurrent context mechanism is particularly attractive because long-form TTS research has demonstrated the usefulness of cached recurrence for propagating global context without processing an entire long document with quadratic attention.

---

# 9. Authored Performance Controls

Automatic performance must be the default.

Manual direction remains available.

Example:

```text
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

Controls SHALL compile into the same representation generated by the automatic planner.

There is therefore no separate “manual mode.”

Human direction simply overrides selected variables.

---

# 10. Acoustic Model

MVE-1 SHOULD use an explicitly duration-regulated acoustic architecture.

Recommended logical pipeline:

```text
phoneme IDs
    ↓
embedding
    ↓
context encoder
    ↓
performance conditioning
    ↓
duration / F0 / energy prediction
    ↓
length regulator
    ↓
frame decoder
    ↓
acoustic representation
```

FastSpeech-family systems demonstrated the practical advantages of explicit duration regulation and prosodic conditioning for controllable parallel synthesis.

MVE does not need to copy those architectures.

It adopts the useful separation of variables.

---

# 11. Proposed Acoustic Representation

Initial target:

```text
sample rate       24,000 Hz
FFT               1024
hop                256 samples
frame rate         93.75 Hz
mel channels       100
```

Each acoustic frame contains:

```text
100 × log-mel coefficients
F0
voicing probability
energy
performance-state projection
```

These values remain configurable at compile/model-generation time rather than being hard-coded throughout the runtime.

---

# 12. Alignment

A first-party training system cannot depend upon an external forced aligner.

MVE therefore SHALL learn monotonic alignment between text phonemes and acoustic frames.

Training computes an alignment probability lattice:

```text
phonemes × acoustic frames
```

and constrains the optimal path to be monotonic.

From that path the engine derives:

```text
phoneme duration
word duration
phrase boundary
pause duration
```

The resulting targets train the standalone duration predictor used at inference.

The final renderer therefore requires no alignment search.

---

# 13. Vocoder

The final high-fidelity system SHALL contain a learned first-party waveform generator.

Input:

```text
mel spectrum
F0
voicing
energy
```

Output:

```text
PCM waveform
```

The initial architecture should favor:

* convolution
* residual blocks
* explicit periodic information
* parallel waveform generation
* CPU-efficient inference

Research such as HiFi-GAN demonstrates that compact convolutional neural vocoders can produce high-quality speech while achieving faster-than-real-time CPU inference in reduced configurations.

MVE SHALL implement its own generator, discriminators, losses and kernels.

The paper is architectural evidence, not a software dependency.

---

# 14. Vocoder Training Loss

The initial loss family should combine:

```math
L =
λ_wave L_wave
+
λ_stft L_stft
+
λ_mel L_mel
+
λ_feat L_feature
+
λ_adv L_adversarial
```

where:

* `L_wave` measures local waveform error where useful.
* `L_stft` evaluates spectral reconstruction at multiple resolutions.
* `L_mel` preserves perceptually relevant spectral structure.
* `L_feature` compares internal discriminator representations.
* `L_adversarial` encourages realistic fine-scale waveform structure.

Training can begin without adversarial components for stability and add them after the generator produces intelligible speech.

---

# 15. Corpus

The training corpus SHALL consist primarily of recordings owned by the speaker.

The recording environment should remain stable:

```text
same microphone
same preamp/interface
same microphone distance
same room
same gain
same sample format
```

Capture target:

```text
48 kHz
24-bit
mono PCM WAV
```

The preprocessing pipeline can generate the internal training sample rate afterward.

---

# 16. Corpus Scale

Recommended engineering targets:

```text
~5 hours
bootstrap / architecture validation

~15–25 hours
first serious single-speaker system

~30–60 hours
target expressive monologue corpus

60+ hours
continued coverage / refinement
```

These are engineering targets rather than guarantees.

For context, the widely used single-speaker LJSpeech corpus contains approximately 24 hours of speech, which has been sufficient for many research TTS systems.

Our requirement is more demanding because the target is not simply readable speech.

We want expressive long-form performance.

---

# 17. Corpus Composition

The corpus SHOULD deliberately contain:

```text
neutral narration
quiet narration
reflective narration
technical explanation
slow deliberate speech
high conviction
rising intensity
controlled anger
warm encouragement
questions
short statements
long sentences
numbers
acronyms
technical vocabulary
intentional silence
breaths
emotional transitions
multi-minute passages
```

Critically, the dataset must contain **actual long passages**.

Training exclusively on isolated sentences would teach sentence-level speech and then ask the system to invent monologue-level behavior at inference.

That is precisely what MVE is designed to avoid.

---

# 18. Recording Unit

The corpus should contain two classes of material.

### Coverage recordings

Shorter controlled utterances designed for:

```text
phonetic coverage
rare sounds
numbers
names
technical terms
stress patterns
```

### Performance recordings

Continuous recordings approximately:

```text
2–15 minutes
```

with exact transcripts.

Performance recordings teach:

```text
cross-sentence pacing
breathing
cadence
long pauses
emotional development
paragraph transitions
section transitions
```

---

# 19. Dataset Integrity

Raw recordings SHALL be immutable.

Suggested structure:

```text
corpus/
    raw/
    transcript/
    manifest/
    derived/
    split/
```

Every source recording receives:

```text
SHA-256
recording-session ID
microphone-chain ID
script ID
take ID
timestamp
sample format
```

Derived artifacts never overwrite originals.

Train, validation and test divisions SHOULD be separated by recording session and source passage rather than simply shuffling adjacent clips.

---

# 20. DSP Runtime

MVE provides first-party implementations of:

```text
RIFF/WAV reader
RIFF/WAV writer
sample conversion
resampler
FFT
inverse FFT
STFT
inverse STFT
window functions
mel filters
F0 extraction
energy extraction
silence detection
normalization
spectral metrics
```

No `libsndfile`, `ffmpeg`, FFTW or equivalent library is part of the canonical path.

Compressed input formats are initially unnecessary.

Canonical source material is WAV.

---

# 21. Neural Runtime

The minimal tensor engine requires:

```text
tensor creation/view
matrix multiplication
embedding lookup
Conv1D
transposed Conv1D
elementwise arithmetic
reductions
normalization
softmax
SiLU / GELU
gated recurrence
dropout during training
upsampling
loss kernels
backpropagation
optimizer state
```

Training begins in `fp32`.

Optimization begins with an internally implemented AdamW-class optimizer.

Later work may introduce:

```text
bf16 training
int8 inference
mixed precision
SIMD kernels
architecture-specific kernels
```

Correctness precedes optimization.

---

# 22. Memory Discipline

The canonical implementation SHOULD follow strict ownership.

A dedicated allocator translation unit owns dynamic allocation.

Subsystems receive arenas or explicit buffers.

Hot synthesis paths SHOULD avoid general-purpose allocation.

Example lifetime classes:

```text
MODEL
persistent for process lifetime

DOCUMENT
persistent for current monologue

UTTERANCE
reset between synthesis units

FRAME
scratch / arena

KERNEL
thread-local scratch
```

This makes memory behavior inspectable and eventually permits aggressive cache optimization.

---

# 23. Threading

CPU parallelism SHALL use deterministic pthread scheduling.

Useful independent regions include:

```text
acoustic frame blocks
convolution channels
vocoder blocks
FFT batches
training minibatches
gradient reductions
feature extraction
```

Given identical:

```text
model
input
seed
build
hardware mode
```

the deterministic runtime SHOULD produce bit-identical output where floating-point execution permits it.

A strict deterministic mode is mandatory even if an optimized relaxed mode is later provided.

---

# 24. Performance Timeline

The second primary output of synthesis is:

```text
performance.mvt
```

The authoritative timebase is audio sample position.

Never floating-point wall-clock time.

Example event:

```c
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
```

This is the in-memory representation only.

The serialized representation SHALL explicitly define field order, width and endian encoding and SHALL NOT serialize compiler-dependent C structure padding.

---

# 25. Timeline Event Types

The timeline supports events including:

```text
PHONEME
WORD
PHRASE
SENTENCE
PARAGRAPH
SECTION

PAUSE_BEGIN
PAUSE_END

BREATH

EMPHASIS_BEGIN
EMPHASIS_END

STATE_KNOT
CONTROL_OVERRIDE
```

Continuous variables can be represented as interpolation knots rather than emitting an event every audio sample.

---

# 26. Why the Timeline Matters

Future engines SHALL consume this file directly.

For example:

```text
                     performance.mvt
                            │
             ┌──────────────┴──────────────┐
             ▼                             ▼
       MUSIC ENGINE                   WISP ENGINE
             │                             │
     harmonic tension                 luminosity
     melodic density                  coherence
     cadence                          color
     dynamic swell                    motion
```

The visual engine never has to guess when a word begins.

The music engine never has to detect where the speaker intended a dramatic pause.

The voice engine already possesses that information.

---

# 27. Model Serialization

MVE model files SHALL use a first-party binary container.

Example:

```text
MVE1
│
├── HEADER
├── MODEL METADATA
├── TENSOR DIRECTORY
├── TENSORS
├── TOKEN TABLE
├── PHONEME TABLE
├── NORMALIZATION DATA
├── PRONUNCIATION DATA
└── CHECKSUMS
```

Each tensor records:

```text
name
dtype
rank
dimensions
byte offset
byte length
checksum
```

No pickle-like executable serialization is permitted.

Loading a model must never execute data-supplied code.

---

# 28. Training Pipeline

The recommended progression is:

```text
PHASE 0
DSP and corpus verification

PHASE 1
monotonic alignment

PHASE 2
duration / pitch / energy predictors

PHASE 3
acoustic decoder

PHASE 4
bootstrap vocoder

PHASE 5
high-fidelity vocoder

PHASE 6
explicit manual performance control

PHASE 7
automatic macro performance planner

PHASE 8
long-form joint refinement
```

Each stage must produce independently inspectable artifacts.

Avoid an architecture where a bad final waveform gives no indication which subsystem failed.

---

# 29. CLI

The entire system SHOULD be operable from one first-party terminal executable.

Conceptually:

```bash
mve corpus verify corpus/

mve preprocess corpus/ build/data/

mve train acoustic config.mve

mve train vocoder config.mve

mve train planner config.mve

mve render monologue.txt \
    --model danny.mve \
    --out episode/

mve inspect episode/performance.mvt

mve bench danny.mve

mve verify danny.mve
```

A successful render produces:

```text
episode/
    narration.wav
    performance.mvt
    words.txt
    render.meta
```

Later:

```text
episode/
    narration.wav
    score.wav
    performance.mvt
    video.frames
    episode.mp4
```

---

# 30. Rendering Boundary

The voice engine SHALL stop at:

```text
narration.wav
performance.mvt
```

It does not initially contain:

* particle rendering
* music generation
* video encoding
* publishing
* YouTube integration

Those systems consume MVE output.

This boundary keeps the vocal engine independently testable.

---

# 31. Quality Gates

## Gate A — intelligibility

No skipped, repeated or hallucinated words in controlled test scripts.

Pronunciation overrides function deterministically.

---

## Gate B — identity

Blind listening tests must consistently identify the synthetic speaker as the intended voice rather than merely a similar voice.

---

## Gate C — short-form naturalness

Thirty-second samples must remain free of obvious:

```text
metallic artifacts
pitch instability
buzzing
over-smoothing
unnatural pauses
cadence loops
```

---

## Gate D — long-form continuity

A ten-minute render must maintain:

```text
speaker identity
loudness consistency
prosodic variation
narrative pacing
pronunciation consistency
```

without obvious sentence-reset behavior.

---

## Gate E — control

Explicit controls must produce predictable changes.

Examples:

```text
@pause 1000ms
```

must create approximately the requested silence.

Increasing intensity must create a monotonic perceptual change rather than an unrelated voice mutation.

---

## Gate F — determinism

Identical deterministic renders SHALL produce identical output hashes under the defined execution environment.

---

## Gate G — speed

Initial target:

```text
RTF < 1.0
```

on a modern high-performance desktop CPU.

Meaning:

```text
10 minutes of audio
<
10 minutes synthesis
```

Second target:

```text
RTF < 0.25
```

or better.

These are engineering objectives, not assumed capabilities.

---

# 32. Parameter Envelope

MVE should begin relatively small.

Suggested research envelope:

```text
acoustic model      10–30M parameters
planner              1–10M
vocoder             10–40M

total target         ~25–75M parameters
```

There is no benefit in making the model large merely because larger models are fashionable.

The domain is extremely constrained:

```text
one speaker
one language initially
one major use case
known recording environment
known performance style
```

Specialization should be exploited.

---

# 33. Inference Memory Target

Initial fp32 target:

```text
< 512 MB model working set
```

Later quantized target:

```text
< 128 MB
```

with frequently executed structures designed to benefit from the CPU cache hierarchy.

The exact limits are benchmark targets and may change after perceptual testing.

Fidelity takes precedence over arbitrary parameter-count reduction.

---

# 34. Architecture Decisions We Explicitly Reject

MVE-1 does not initially pursue:

```text
universal zero-shot voice cloning
thousands of speakers
speech-to-speech conversion
multilingual synthesis
giant codec language models
mandatory diffusion sampling
cloud synthesis
sentence-isolated rendering
opaque automatic emotion only
```

Those problems consume capacity without directly serving the objective.

---

# 35. Research Basis Versus MVE Design

Several established results motivate parts of the system:

**Explicit prosody:** duration, pitch and energy are useful controllable conditioning variables.

**Stochastic rhythm:** systems such as VITS demonstrate that duration itself contains meaningful expressive variation rather than being a fixed linguistic quantity.

**Neural waveform synthesis:** high-quality parallel vocoding is practical and can be computationally efficient.

**Cross-sentence context:** paragraph-oriented TTS research reports advantages from retaining linguistic and prosodic information between sentences.

These establish useful engineering principles.

They do **not** establish that the complete MVE architecture described here has been validated.

That must be demonstrated experimentally.

Nor should MVE claim architectural novelty until a proper literature and patent review is performed.

---

# 36. First Prototype

The first implementation should be much smaller than the final architecture.

### MVE-0

```text
WAV reader/writer
FFT/STFT/mel
corpus manifest
text normalization
phoneme representation
alignment
duration predictor
pitch predictor
energy predictor
small acoustic network
bootstrap waveform synthesis
timeline writer
CLI
```

Success:

> A typed paragraph becomes recognizable, intelligible speech in the target voice entirely through our C runtime.

Nothing else matters yet.

---

# 37. MVE-0.5

Next:

```text
learned neural vocoder
better pronunciation
manual performance controls
phrase-level continuity
breath representation
```

Success:

> One minute of speech sounds convincingly like a deliberate recording.

---

# 38. MVE-1

Then:

```text
macro planner
paragraph context
section context
performance trajectories
long-form dataset
long-form training objectives
high-fidelity vocoder
CPU optimization
deterministic timeline
```

Success:

> Ten-minute monologues remain convincing as continuous performances.

---

# 39. Future System

Once MVE-1 is stable:

```text
                         SCRIPT
                            │
                            ▼
                         MVE-1
                            │
                 ┌──────────┴──────────┐
                 ▼                     ▼
          narration.wav         performance.mvt
                                       │
                           ┌───────────┴───────────┐
                           ▼                       ▼
                    MELODY ENGINE              WISP
                           │                       │
                           └───────────┬───────────┘
                                       ▼
                              MEDIA RENDERER
                                       │
                                       ▼
                                 FINAL FILM
```

The music and visual systems therefore do not merely respond to an audio waveform.

They respond to the same underlying performance state that produced the voice.

---

# 40. Final Design Thesis

MVE is based on a simple premise:

> **A voice is not merely a timbre. It is a performance through time.**

Speaker cloning solves only part of the problem.

For long-form monologue work, the system must preserve:

```text
identity
+
language
+
timing
+
prosody
+
emotion
+
continuity
+
directorial intent
```

The performance timeline makes those properties explicit.

That creates a foundation larger than TTS itself.

The synthesized voice becomes the temporal authority for a future generative media system in which narration, melody, sound, light, motion and color are generated from one coherent performance.

MVE-1 is therefore not designed merely as a text-to-speech engine.

It is the first component of a **first-party digital performance runtime**.

---

## Canonical Definition

**MVE-1 is a proprietary, single-speaker, long-form monologue synthesis engine implemented end-to-end in C23, trained on first-party recordings, providing explicit prosodic and narrative performance control, deterministic local inference, high-fidelity waveform generation, and a sample-accurate machine-readable performance timeline as a first-class output.**
