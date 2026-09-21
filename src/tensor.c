/* MVE-1 First-Party Tensor Engine */
#include "mve_tensor.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

static size_t compute_numel(int ndim, const size_t* shape) {
    size_t n = 1;
    for (int i = 0; i < ndim; i++) n *= shape[i];
    return n;
}

static void compute_strides(int ndim, const size_t* shape, size_t* strides) {
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--)
        strides[i] = strides[i + 1] * shape[i + 1];
}

MVE_Tensor* mve_tensor_create(void) {
    MVE_Tensor* t = calloc(1, sizeof(MVE_Tensor));
    if (t) { t->owns_data = 1; t->dtype = MVE_DTYPE_FLOAT32; }
    return t;
}

MVE_Tensor* mve_tensor_create_named(const char* name) {
    MVE_Tensor* t = mve_tensor_create();
    if (t && name) strncpy(t->name, name, MVE_TENSOR_MAX_NAME_LEN - 1);
    return t;
}

MVE_Tensor* mve_tensor_zeros(int ndim, const size_t* shape) {
    if (ndim <= 0 || ndim > MVE_TENSOR_MAX_DIMS) return NULL;
    MVE_Tensor* t = mve_tensor_create();
    if (!t) return NULL;
    t->ndim = ndim;
    memcpy(t->shape, shape, ndim * sizeof(size_t));
    compute_strides(ndim, shape, t->strides);
    size_t numel = compute_numel(ndim, shape);
    t->data = calloc(numel, sizeof(float));
    if (!t->data) { free(t); return NULL; }
    return t;
}

MVE_Tensor* mve_tensor_ones(int ndim, const size_t* shape) {
    MVE_Tensor* t = mve_tensor_zeros(ndim, shape);
    if (!t) return NULL;
    size_t n = compute_numel(ndim, shape);
    for (size_t i = 0; i < n; i++) t->data[i] = 1.0f;
    return t;
}

MVE_Tensor* mve_tensor_empty(int ndim, const size_t* shape) {
    if (ndim <= 0 || ndim > MVE_TENSOR_MAX_DIMS) return NULL;
    MVE_Tensor* t = mve_tensor_create();
    if (!t) return NULL;
    t->ndim = ndim;
    memcpy(t->shape, shape, ndim * sizeof(size_t));
    compute_strides(ndim, shape, t->strides);
    size_t numel = compute_numel(ndim, shape);
    t->data = malloc(numel * sizeof(float));
    if (!t->data) { free(t); return NULL; }
    return t;
}

MVE_Tensor* mve_tensor_from_data(const float* data, int ndim, const size_t* shape) {
    MVE_Tensor* t = mve_tensor_empty(ndim, shape);
    if (!t) return NULL;
    memcpy(t->data, data, compute_numel(ndim, shape) * sizeof(float));
    return t;
}

void mve_tensor_destroy(MVE_Tensor* t) {
    if (!t) return;
    if (t->owns_data && t->data) free(t->data);
    free(t);
}

size_t mve_tensor_numel(const MVE_Tensor* t) {
    return t ? compute_numel(t->ndim, t->shape) : 0;
}

size_t mve_tensor_nbytes(const MVE_Tensor* t) {
    return mve_tensor_numel(t) * sizeof(float);
}

int mve_tensor_reshape(MVE_Tensor* t, int ndim, const size_t* shape) {
    if (!t || ndim <= 0 || ndim > MVE_TENSOR_MAX_DIMS) return -1;
    if (compute_numel(ndim, shape) != mve_tensor_numel(t)) return -1;
    t->ndim = ndim;
    memcpy(t->shape, shape, ndim * sizeof(size_t));
    compute_strides(ndim, shape, t->strides);
    return 0;
}

int mve_tensor_flatten(MVE_Tensor* t) {
    if (!t) return -1;
    t->ndim = 1;
    t->shape[0] = mve_tensor_numel(t);
    t->strides[0] = 1;
    return 0;
}

float mve_tensor_get(const MVE_Tensor* t, const size_t* indices) {
    if (!t || !indices) return 0.0f;
    size_t off = 0;
    for (int i = 0; i < t->ndim; i++) off += indices[i] * t->strides[i];
    return t->data[off];
}

int mve_tensor_set(MVE_Tensor* t, const size_t* indices, float v) {
    if (!t || !indices) return -1;
    size_t off = 0;
    for (int i = 0; i < t->ndim; i++) {
        if (indices[i] >= t->shape[i]) return -1;
        off += indices[i] * t->strides[i];
    }
    t->data[off] = v;
    return 0;
}

int mve_tensor_fill(MVE_Tensor* t, float v) {
    if (!t) return -1;
    size_t n = mve_tensor_numel(t);
    for (size_t i = 0; i < n; i++) t->data[i] = v;
    return 0;
}

int mve_tensor_zero(MVE_Tensor* t) {
    if (!t) return -1;
    memset(t->data, 0, mve_tensor_nbytes(t));
    return 0;
}

int mve_tensor_copy(MVE_Tensor* dst, const MVE_Tensor* src) {
    if (!dst || !src) return -1;
    if (mve_tensor_numel(dst) < mve_tensor_numel(src)) return -1;
    memcpy(dst->data, src->data, mve_tensor_numel(src) * sizeof(float));
    return 0;
}

MVE_Tensor* mve_tensor_clone(const MVE_Tensor* t) {
    if (!t) return NULL;
    MVE_Tensor* c = mve_tensor_empty(t->ndim, t->shape);
    if (!c) return NULL;
    mve_tensor_copy(c, t);
    strncpy(c->name, t->name, MVE_TENSOR_MAX_NAME_LEN - 1);
    return c;
}

MVE_Tensor* mve_tensor_view(const MVE_Tensor* t, int ndim, const size_t* shape, const size_t* strides) {
    if (!t || ndim <= 0 || ndim > MVE_TENSOR_MAX_DIMS) return NULL;
    MVE_Tensor* v = mve_tensor_create();
    if (!v) return NULL;
    v->ndim = ndim;
    memcpy(v->shape, shape, ndim * sizeof(size_t));
    memcpy(v->strides, strides, ndim * sizeof(size_t));
    v->data = t->data;
    v->owns_data = 0;
    v->dtype = t->dtype;
    return v;
}

MVE_Tensor* mve_tensor_slice(const MVE_Tensor* t, int dim, size_t start, size_t end) {
    if (!t || dim < 0 || dim >= t->ndim) return NULL;
    if (start >= t->shape[dim] || end > t->shape[dim] || start >= end) return NULL;
    size_t ns[MVE_TENSOR_MAX_DIMS], st[MVE_TENSOR_MAX_DIMS];
    memcpy(ns, t->shape, t->ndim * sizeof(size_t));
    memcpy(st, t->strides, t->ndim * sizeof(size_t));
    ns[dim] = end - start;
    MVE_Tensor* s = mve_tensor_create();
    if (!s) return NULL;
    s->ndim = t->ndim;
    memcpy(s->shape, ns, t->ndim * sizeof(size_t));
    memcpy(s->strides, st, t->ndim * sizeof(size_t));
    s->data = t->data + start * t->strides[dim];
    s->owns_data = 0;
    s->dtype = t->dtype;
    return s;
}

MVE_Matrix* mve_matrix_create(int r, int c) {
    if (r <= 0 || c <= 0) return NULL;
    MVE_Matrix* m = malloc(sizeof(MVE_Matrix));
    if (!m) return NULL;
    m->rows = r; m->cols = c;
    m->data = malloc(r * c * sizeof(float));
    m->owns_data = 1;
    if (!m->data) { free(m); return NULL; }
    return m;
}

MVE_Matrix* mve_matrix_zeros(int r, int c) {
    MVE_Matrix* m = mve_matrix_create(r, c);
    if (m) memset(m->data, 0, r * c * sizeof(float));
    return m;
}

MVE_Matrix* mve_matrix_eye(int n) {
    MVE_Matrix* m = mve_matrix_zeros(n, n);
    if (m) for (int i = 0; i < n; i++) m->data[i * n + i] = 1.0f;
    return m;
}

void mve_matrix_destroy(MVE_Matrix* m) {
    if (!m) return;
    if (m->owns_data && m->data) free(m->data);
    free(m);
}

int mve_matrix_fill(MVE_Matrix* m, float v) {
    if (!m) return -1;
    int n = m->rows * m->cols;
    for (int i = 0; i < n; i++) m->data[i] = v;
    return 0;
}

int mve_matrix_zero(MVE_Matrix* m) {
    if (!m) return -1;
    memset(m->data, 0, m->rows * m->cols * sizeof(float));
    return 0;
}

int mve_matrix_copy(MVE_Matrix* d, const MVE_Matrix* s) {
    if (!d || !s) return -1;
    if (d->rows != s->rows || d->cols != s->cols) return -1;
    memcpy(d->data, s->data, s->rows * s->cols * sizeof(float));
    return 0;
}

int mve_matrix_gemm(const MVE_Matrix* A, const MVE_Matrix* B, MVE_Matrix* C,
                    int tA, int tB, float alpha, float beta) {
    if (!A || !B || !C) return -1;
    int m = tA ? A->cols : A->rows;
    int ka = tA ? A->rows : A->cols;
    int kb = tB ? B->cols : B->rows;
    int n = tB ? B->rows : B->cols;
    if (ka != kb || C->rows != m || C->cols != n) return -1;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0;
            for (int k = 0; k < ka; k++) {
                float a = tA ? A->data[k * A->rows + i] : A->data[i * A->cols + k];
                float b = tB ? B->data[j * B->cols + k] : B->data[k * B->cols + j];
                sum += a * b;
            }
            C->data[i * C->cols + j] = beta * C->data[i * C->cols + j] + alpha * sum;
        }
    }
    return 0;
}

int mve_matrix_gemv(const MVE_Matrix* A, const float* x, float* y, int t, float alpha, float beta) {
    if (!A || !x || !y) return -1;
    int m = t ? A->cols : A->rows;
    int n = t ? A->rows : A->cols;
    for (int i = 0; i < m; i++) {
        float sum = 0;
        for (int j = 0; j < n; j++) {
            float a = t ? A->data[j * A->cols + i] : A->data[i * A->cols + j];
            sum += a * x[j];
        }
        y[i] = beta * y[i] + alpha * sum;
    }
    return 0;
}

int mve_matrix_add(MVE_Matrix* C, const MVE_Matrix* A, const MVE_Matrix* B) {
    if (!C || !A || !B) return -1;
    if (C->rows != A->rows || C->cols != A->cols || C->rows != B->rows || C->cols != B->cols) return -1;
    int n = A->rows * A->cols;
    for (int i = 0; i < n; i++) C->data[i] = A->data[i] + B->data[i];
    return 0;
}

int mve_matrix_sub(MVE_Matrix* C, const MVE_Matrix* A, const MVE_Matrix* B) {
    if (!C || !A || !B) return -1;
    if (C->rows != A->rows || C->cols != A->cols || C->rows != B->rows || C->cols != B->cols) return -1;
    int n = A->rows * A->cols;
    for (int i = 0; i < n; i++) C->data[i] = A->data[i] - B->data[i];
    return 0;
}

int mve_matrix_mul(MVE_Matrix* C, const MVE_Matrix* A, const MVE_Matrix* B) {
    if (!C || !A || !B) return -1;
    if (C->rows != A->rows || C->cols != A->cols || C->rows != B->rows || C->cols != B->cols) return -1;
    int n = A->rows * A->cols;
    for (int i = 0; i < n; i++) C->data[i] = A->data[i] * B->data[i];
    return 0;
}

int mve_matrix_scale(MVE_Matrix* A, float s) {
    if (!A) return -1;
    int n = A->rows * A->cols;
    for (int i = 0; i < n; i++) A->data[i] *= s;
    return 0;
}

MVE_Arena* mve_arena_create(size_t initial) {
    if (initial == 0) initial = MVE_ARENA_BLOCK_SIZE;
    MVE_ArenaBlock* blk = malloc(sizeof(MVE_ArenaBlock));
    if (!blk) return NULL;
    blk->data = malloc(initial);
    if (!blk->data) { free(blk); return NULL; }
    blk->size = initial; blk->used = 0; blk->next = NULL;
    MVE_Arena* ar = malloc(sizeof(MVE_Arena));
    if (!ar) { free(blk->data); free(blk); return NULL; }
    ar->first = ar->current = blk;
    ar->total_allocated = initial; ar->peak_used = 0;
    return ar;
}

void mve_arena_destroy(MVE_Arena* ar) {
    if (!ar) return;
    for (MVE_ArenaBlock* b = ar->first; b;) {
        MVE_ArenaBlock* n = b->next;
        free(b->data); free(b); b = n;
    }
    free(ar);
}

void* mve_arena_alloc(MVE_Arena* ar, size_t size) {
    if (!ar || size == 0) return NULL;
    size = (size + 7) & ~7;
    MVE_ArenaBlock* b = ar->current;
    if (b->used + size <= b->size) {
        void* p = b->data + b->used;
        b->used += size;
        size_t tot = 0;
        for (MVE_ArenaBlock* x = ar->first; x; x = x->next) tot += x->used;
        if (tot > ar->peak_used) ar->peak_used = tot;
        return p;
    }
    size_t ns = (size > MVE_ARENA_BLOCK_SIZE) ? size : MVE_ARENA_BLOCK_SIZE;
    MVE_ArenaBlock* nb = malloc(sizeof(MVE_ArenaBlock));
    if (!nb) return NULL;
    nb->data = malloc(ns);
    if (!nb->data) { free(nb); return NULL; }
    nb->size = ns; nb->used = size; nb->next = NULL;
    b->next = nb; ar->current = nb;
    ar->total_allocated += ns; ar->peak_used += size;
    return nb->data;
}

void* mve_arena_alloc_aligned(MVE_Arena* ar, size_t size, size_t align) {
    if (!ar || size == 0 || align == 0) return NULL;
    return mve_arena_alloc(ar, ((size + align - 1) / align) * align);
}

void mve_arena_reset(MVE_Arena* ar) {
    if (!ar) return;
    for (MVE_ArenaBlock* b = ar->first; b; b = b->next) b->used = 0;
    ar->current = ar->first;
}

void mve_arena_checkpoint(MVE_Arena* ar, size_t* cp) {
    if (!ar || !cp) return;
    size_t idx = 0;
    for (MVE_ArenaBlock* b = ar->first; b && b != ar->current; b = b->next) idx++;
    cp[0] = idx; cp[1] = ar->current->used;
}

void mve_arena_restore(MVE_Arena* ar, size_t* cp) {
    if (!ar || !cp) return;
    MVE_ArenaBlock* b = ar->first;
    while (cp[0] > 0 && b) { b = b->next; cp[0]--; }
    if (b) {
        ar->current = b; b->used = cp[1];
        for (MVE_ArenaBlock* n = b->next; n; n = n->next) n->used = 0;
    }
}

int mve_activator_relu(float* d, size_t n) { if (!d) return -1; for (size_t i = 0; i < n; i++) if (d[i] < 0) d[i] = 0; return 0; }
int mve_activator_gelu(float* d, size_t n) {
    if (!d) return -1;
    const float c = 0.7978845608028654f;
    for (size_t i = 0; i < n; i++) { float x = d[i], x3 = x*x*x; d[i] = 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x3))); }
    return 0;
}
int mve_activator_silu(float* d, size_t n) {
    if (!d) return -1;
    for (size_t i = 0; i < n; i++) { float x = d[i]; d[i] = x / (1.0f + expf(-x)); }
    return 0;
}
int mve_activator_tanh(float* d, size_t n) { if (!d) return -1; for (size_t i = 0; i < n; i++) d[i] = tanhf(d[i]); return 0; }
int mve_activator_sigmoid(float* d, size_t n) { if (!d) return -1; for (size_t i = 0; i < n; i++) d[i] = 1.0f / (1.0f + expf(-d[i])); return 0; }

int mve_layernorm_forward(const float* in, float* out, const float* w, const float* bias, int n, int hs, float eps) {
    if (!in || !out || n <= 0 || hs <= 0) return -1;
    int inst = n / hs;
    for (int i = 0; i < inst; i++) {
        const float* x = in + i * hs;
        float* y = out + i * hs;
        float mean = 0, var = 0;
        for (int j = 0; j < hs; j++) mean += x[j];
        mean /= hs;
        for (int j = 0; j < hs; j++) { float d = x[j] - mean; var += d * d; }
        var /= hs;
        float is = 1.0f / sqrtf(var + eps);
        for (int j = 0; j < hs; j++) {
            float v = (x[j] - mean) * is;
            if (w) v *= w[j];
            if (bias) v += bias[j];
            y[j] = v;
        }
    }
    return 0;
}

const char* mve_dtype_string(MVE_DType dt) {
    switch(dt) { case MVE_DTYPE_FLOAT32: return "float32"; case MVE_DTYPE_FLOAT16: return "float16";
        case MVE_DTYPE_INT8: return "int8"; case MVE_DTYPE_INT32: return "int32"; default: return "unknown"; }
}
size_t mve_dtype_size(MVE_DType dt) {
    switch(dt) { case MVE_DTYPE_FLOAT32: return 4; case MVE_DTYPE_FLOAT16: return 2;
        case MVE_DTYPE_INT8: return 1; case MVE_DTYPE_INT32: return 4; default: return 0; }
}

void mve_print_tensor(const MVE_Tensor* t, const char* nm) {
    if (!t) { printf("%s: (null)\n", nm ? nm : "tensor"); return; }
    printf("%s: shape=(", nm ? nm : "tensor");
    for (int i = 0; i < t->ndim; i++) { printf("%zu", t->shape[i]); if (i < t->ndim-1) printf(", "); }
    printf("), dtype=%s, numel=%zu\n", mve_dtype_string(t->dtype), mve_tensor_numel(t));
    size_t n = mve_tensor_numel(t), pc = (n < 10) ? n : 10;
    printf("  [");
    for (size_t i = 0; i < pc; i++) { printf("%.4f", t->data[i]); if (i < pc-1) printf(", "); }
    if (n > pc) printf(", ... (%zu more)", n-pc);
    printf("]\n");
}

void mve_print_layer_stats(const MVE_Layer* l) {
    if (!l) { printf("layer: (null)\n"); return; }
    printf("Layer '%s': type=%d", l->name, l->type);
    if (l->weight) {
        printf(", weight_shape=(");
        for (int i = 0; i < l->weight->ndim; i++) { printf("%zu", l->weight->shape[i]); if (i < l->weight->ndim-1) printf(", "); }
        printf(")");
    }
    printf("\n");
}
