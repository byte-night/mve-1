#ifndef MVE_TENSOR_H
#define MVE_TENSOR_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * MVE-1: First-Party Tensor Engine
 * Minimal neural network substrate for acoustic model, vocoder, and planner
 * ============================================================================ */

#define MVE_TENSOR_MAX_DIMS 4
#define MVE_TENSOR_MAX_NAME_LEN 64

/* Data types */
typedef enum {
    MVE_DTYPE_FLOAT32 = 0,
    MVE_DTYPE_FLOAT16,
    MVE_DTYPE_INT8,
    MVE_DTYPE_INT32
} MVE_DType;

/* Tensor structure - owns its data */
typedef struct {
    float* data;              /* Row-major contiguous data */
    size_t shape[MVE_TENSOR_MAX_DIMS];
    size_t strides[MVE_TENSOR_MAX_DIMS];
    int ndim;
    MVE_DType dtype;
    char name[MVE_TENSOR_MAX_NAME_LEN];
    int owns_data;            /* Whether this tensor owns the memory */
} MVE_Tensor;

/* Matrix for 2D operations */
typedef struct {
    float* data;
    int rows;
    int cols;
    int owns_data;
} MVE_Matrix;

/* ============================================================================
 * Memory Arena - for scratch allocation during inference
 * ============================================================================ */

#define MVE_ARENA_BLOCK_SIZE (1024 * 1024)  /* 1MB blocks */

typedef struct MVE_ArenaBlock {
    uint8_t* data;
    size_t size;
    size_t used;
    struct MVE_ArenaBlock* next;
} MVE_ArenaBlock;

typedef struct {
    MVE_ArenaBlock* first;
    MVE_ArenaBlock* current;
    size_t total_allocated;
    size_t peak_used;
} MVE_Arena;

/* ============================================================================
 * Neural Network Layer Types
 * ============================================================================ */

typedef enum {
    MVE_LAYER_LINEAR = 0,
    MVE_LAYER_CONV1D,
    MVE_LAYER_CONVTRANSPOSE1D,
    MVE_LAYER_EMBEDDING,
    MVE_LAYER_LAYERNORM,
    MVE_LAYER_GRU,
    MVE_LAYER_LSTM,
    MVE_LAYER_ATTENTION,
    MVE_LAYER_RESIDUAL,
    MVE_LAYER_UPSAMPLE,
    MVE_LAYER_ACTIVATION
} MVE_LayerType;

typedef enum {
    MVE_ACT_RELU = 0,
    MVE_ACT_GELU,
    MVE_ACT_SILU,
    MVE_ACT_TANH,
    MVE_ACT_SIGMOID
} MVE_Activation;

/* Base layer structure */
typedef struct MVE_Layer {
    MVE_LayerType type;
    char name[MVE_TENSOR_MAX_NAME_LEN];
    
    /* Weights */
    MVE_Tensor* weight;
    MVE_Tensor* bias;
    
    /* State for recurrent layers */
    MVE_Tensor* hidden_state;
    MVE_Tensor* cell_state;
    
    /* Layer-specific parameters */
    int in_features;
    int out_features;
    int kernel_size;
    int stride;
    int padding;
    int dilation;
    int groups;
    
    MVE_Activation activation;
    
    /* Gradient storage for training */
    MVE_Tensor* grad_weight;
    MVE_Tensor* grad_bias;
    MVE_Tensor* grad_input;
    
    struct MVE_Layer* next;
} MVE_Layer;

/* Sequential container */
typedef struct {
    MVE_Layer* first;
    MVE_Layer* last;
    int num_layers;
} MVE_Sequential;

/* ============================================================================
 * Optimizer State
 * ============================================================================ */

typedef enum {
    MVE_OPT_SGD = 0,
    MVE_OPT_ADAM,
    MVE_OPT_ADAMW
} MVE_OptimizerType;

typedef struct {
    MVE_OptimizerType type;
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    float weight_decay;
    
    /* Adam state */
    MVE_Tensor* m;  /* First moment */
    MVE_Tensor* v;  /* Second moment */
    int timestep;
} MVE_Optimizer;

/* ============================================================================
 * Loss Functions
 * ============================================================================ */

typedef enum {
    MVE_LOSS_MSE = 0,
    MVE_LOSS_L1,
    MVE_LOSS_BCE,
    MVE_LOSS_CROSSENTROPY,
    MVE_LOSS_HUBER,
    MVE_LOSS_STFT,
    MVE_LOSS_MEL,
    MVE_LOSS_FEATURE_MATCH,
    MVE_LOSS_ADVERSARIAL
} MVE_LossType;

typedef struct {
    MVE_LossType type;
    float reduction;  /* 'none'=0, 'mean'=1, 'sum'=2 */
    
    /* STFT loss parameters */
    int fft_size;
    int hop_size;
    int win_length;
    float w_log_mag;
    float w_lin_mag;
    
    /* Feature matching */
    int num_scales;
    int* scale_fft_sizes;
} MVE_Loss;

/* ============================================================================
 * Model Container
 * ============================================================================ */

typedef struct {
    char name[64];
    MVE_Sequential layers;
    MVE_Optimizer optimizer;
    MVE_Loss loss;
    
    /* Training state */
    int training;
    int epoch;
    int step;
    float running_loss;
    
    /* Statistics */
    size_t num_parameters;
    size_t parameter_memory_bytes;
} MVE_Network;

/* ============================================================================
 * Tensor API
 * ============================================================================ */

/* Creation/destruction */
MVE_Tensor* mve_tensor_create(void);
MVE_Tensor* mve_tensor_create_named(const char* name);
MVE_Tensor* mve_tensor_zeros(int ndim, const size_t* shape);
MVE_Tensor* mve_tensor_ones(int ndim, const size_t* shape);
MVE_Tensor* mve_tensor_empty(int ndim, const size_t* shape);
MVE_Tensor* mve_tensor_from_data(const float* data, int ndim, const size_t* shape);
void mve_tensor_destroy(MVE_Tensor* t);

/* Shape manipulation */
int mve_tensor_reshape(MVE_Tensor* t, int ndim, const size_t* shape);
int mve_tensor_flatten(MVE_Tensor* t);
size_t mve_tensor_numel(const MVE_Tensor* t);
size_t mve_tensor_nbytes(const MVE_Tensor* t);

/* Data access */
float mve_tensor_get(const MVE_Tensor* t, const size_t* indices);
int mve_tensor_set(MVE_Tensor* t, const size_t* indices, float value);
int mve_tensor_fill(MVE_Tensor* t, float value);
int mve_tensor_zero(MVE_Tensor* t);
int mve_tensor_copy(MVE_Tensor* dst, const MVE_Tensor* src);
MVE_Tensor* mve_tensor_clone(const MVE_Tensor* t);

/* Views (no copy) */
MVE_Tensor* mve_tensor_view(const MVE_Tensor* t, int ndim, const size_t* shape, const size_t* strides);
MVE_Tensor* mve_tensor_slice(const MVE_Tensor* t, int dim, size_t start, size_t end);

/* ============================================================================
 * Matrix Operations
 * ============================================================================ */

MVE_Matrix* mve_matrix_create(int rows, int cols);
MVE_Matrix* mve_matrix_zeros(int rows, int cols);
MVE_Matrix* mve_matrix_eye(int n);
void mve_matrix_destroy(MVE_Matrix* m);

int mve_matrix_fill(MVE_Matrix* m, float value);
int mve_matrix_zero(MVE_Matrix* m);
int mve_matrix_copy(MVE_Matrix* dst, const MVE_Matrix* src);

/* BLAS-like operations */
int mve_matrix_gemm(
    const MVE_Matrix* A,
    const MVE_Matrix* B,
    MVE_Matrix* C,
    int transpose_A,
    int transpose_B,
    float alpha,
    float beta
);

int mve_matrix_gemv(
    const MVE_Matrix* A,
    const float* x,
    float* y,
    int transpose,
    float alpha,
    float beta
);

/* Element-wise */
int mve_matrix_add(MVE_Matrix* C, const MVE_Matrix* A, const MVE_Matrix* B);
int mve_matrix_sub(MVE_Matrix* C, const MVE_Matrix* A, const MVE_Matrix* B);
int mve_matrix_mul(MVE_Matrix* C, const MVE_Matrix* A, const MVE_Matrix* B);
int mve_matrix_scale(MVE_Matrix* A, float s);

/* ============================================================================
 * Arena API
 * ============================================================================ */

MVE_Arena* mve_arena_create(size_t initial_size);
void mve_arena_destroy(MVE_Arena* arena);
void* mve_arena_alloc(MVE_Arena* arena, size_t size);
void* mve_arena_alloc_aligned(MVE_Arena* arena, size_t size, size_t alignment);
void mve_arena_reset(MVE_Arena* arena);
void mve_arena_checkpoint(MVE_Arena* arena, size_t* checkpoint);
void mve_arena_restore(MVE_Arena* arena, size_t* checkpoint);

/* ============================================================================
 * Layer Factory
 * ============================================================================ */

MVE_Layer* mve_layer_linear(int in_features, int out_features, int use_bias);
MVE_Layer* mve_layer_conv1d(int in_channels, int out_channels, int kernel_size, 
                            int stride, int padding, int dilation, int groups);
MVE_Layer* mve_layer_convtranspose1d(int in_channels, int out_channels, int kernel_size,
                                     int stride, int padding, int output_padding);
MVE_Layer* mve_layer_embedding(int num_embeddings, int embedding_dim);
MVE_Layer* mve_layer_layernorm(int normalized_shape, float eps);
MVE_Layer* mve_layer_gru(int input_size, int hidden_size, int num_layers);
MVE_Layer* mve_layer_activation(MVE_Activation act);
MVE_Layer* mve_layer_residual(MVE_Layer* inner);

void mve_layer_destroy(MVE_Layer* layer);

/* ============================================================================
 * Sequential API
 * ============================================================================ */

MVE_Sequential* mve_sequential_create(void);
void mve_sequential_destroy(MVE_Sequential* seq);
int mve_sequential_append(MVE_Sequential* seq, MVE_Layer* layer);

/* Forward pass */
MVE_Tensor* mve_sequential_forward(MVE_Sequential* seq, MVE_Tensor* input);

/* ============================================================================
 * Activation Functions (in-place)
 * ============================================================================ */

int mve_activator_relu(float* data, size_t n);
int mve_activator_gelu(float* data, size_t n);
int mve_activator_silu(float* data, size_t n);
int mve_activator_tanh(float* data, size_t n);
int mve_activator_sigmoid(float* data, size_t n);

/* ============================================================================
 * Normalization
 * ============================================================================ */

int mve_layernorm_forward(const float* input, float* output, const float* weight,
                          const float* bias, int n, int hidden_size, float eps);

/* ============================================================================
 * Convolution Primitives
 * ============================================================================ */

/* 1D convolution: (N, C_in, L_in) -> (N, C_out, L_out) */
int mve_conv1d_forward(
    const float* input,   /* [N, C_in, L_in] */
    float* output,        /* [N, C_out, L_out] */
    const float* weight,  /* [C_out, C_in/groups, K] */
    const float* bias,    /* [C_out] */
    int N, int C_in, int C_out, int L_in, int K,
    int stride, int padding, int dilation, int groups
);

/* Transposed 1D convolution */
int mve_convtranspose1d_forward(
    const float* input,   /* [N, C_in, L_in] */
    float* output,        /* [N, C_out, L_out] */
    const float* weight,  /* [C_in, C_out/groups, K] */
    const float* bias,    /* [C_out] */
    int N, int C_in, int C_out, int L_in, int K,
    int stride, int padding, int output_padding, int groups
);

/* ============================================================================
 * Recurrent Primitives
 * ============================================================================ */

/* GRU forward step */
int mve_gru_forward(
    const float* input,      /* [N, input_size] */
    float* hidden,           /* [N, hidden_size] */
    float* output,           /* [N, hidden_size] */
    const float* weight_ih,  /* [3*hidden, input] */
    const float* weight_hh,  /* [3*hidden, hidden] */
    const float* bias_ih,    /* [3*hidden] */
    const float* bias_hh,    /* [3*hidden] */
    int N, int input_size, int hidden_size
);

/* ============================================================================
 * Attention
 * ============================================================================ */

/* Scaled dot-product attention */
int mve_attention_forward(
    const float* query,     /* [N, H, L_q, D] */
    const float* key,       /* [N, H, L_k, D] */
    const float* value,     /* [N, H, L_v, D] */
    float* output,          /* [N, H, L_q, D] */
    const float* mask,      /* [L_q, L_k] or NULL */
    int N, int H, int L_q, int L_k, int L_v, int D,
    float scale
);

/* ============================================================================
 * Loss Functions
 * ============================================================================ */

float mve_loss_mse(const float* pred, const float* target, size_t n);
float mve_loss_l1(const float* pred, const float* target, size_t n);
float mve_loss_bce(const float* pred, const float* target, size_t n);

/* Multi-resolution STFT loss */
float mve_loss_stft(
    const float* pred, const float* target,
    int n_samples, int sample_rate,
    int fft_size, int hop_size, int win_length,
    float w_log_mag, float w_lin_mag
);

/* ============================================================================
 * Optimizers
 * ============================================================================ */

MVE_Optimizer* mve_optimizer_adamw(float lr, float beta1, float beta2, 
                                   float eps, float weight_decay);
void mve_optimizer_destroy(MVE_Optimizer* opt);

int mve_optimizer_step(MVE_Optimizer* opt, MVE_Layer* layer);
int mve_optimizer_zero_grad(MVE_Optimizer* opt, MVE_Layer* layer);

/* ============================================================================
 * Serialization
 * ============================================================================ */

int mve_network_save(const MVE_Network* net, const char* filepath);
int mve_network_load(MVE_Network* net, const char* filepath);

/* ============================================================================
 * Utilities
 * ============================================================================ */

const char* mve_dtype_string(MVE_DType dtype);
size_t mve_dtype_size(MVE_DType dtype);
void mve_print_tensor(const MVE_Tensor* t, const char* name);
void mve_print_layer_stats(const MVE_Layer* layer);

#ifdef __cplusplus
}
#endif

#endif /* MVE_TENSOR_H */
