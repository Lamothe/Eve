#include "eve_native.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-vulkan.h"

struct weight_entry {
    char name[128];
    struct ggml_tensor * tensor;
};

struct compute_ctx {
    struct ggml_context * ctx;
    struct ggml_cgraph * gf;
    ggml_backend_buffer_t buf;
};

struct eve_handle {
    // Backends
    ggml_backend_t vk_backend;
    ggml_backend_t cpu_backend;
    ggml_backend_buffer_type_t vk_buft;
    ggml_backend_buffer_t weight_buf;

    // Persistent contexts for weights
    struct ggml_context * weight_ctx;
    size_t weight_mem_size;

    // Transformer config
    int vocab_size;
    int embed_dim;
    int num_heads;
    int num_layers;
    int feed_forward_dim;
    int max_seq_len;

    // Transformer weights: token embedding (vocab_size x embed_dim)
    struct ggml_tensor * trans_token_emb;

    // Transformer weights: positional embedding
    struct ggml_tensor * trans_pos_emb;

    // Transformer weights: per-layer (arrays of size num_layers)
    struct ggml_tensor ** trans_attn_norm_w;
    struct ggml_tensor ** trans_attn_norm_b;
    struct ggml_tensor ** trans_wq;
    struct ggml_tensor ** trans_bq;
    struct ggml_tensor ** trans_wk;
    struct ggml_tensor ** trans_bk;
    struct ggml_tensor ** trans_wv;
    struct ggml_tensor ** trans_bv;
    struct ggml_tensor ** trans_wo;
    struct ggml_tensor ** trans_bo;
    struct ggml_tensor ** trans_ffn_norm_w;
    struct ggml_tensor ** trans_ffn_norm_b;
    struct ggml_tensor ** trans_w1;
    struct ggml_tensor ** trans_b1;
    struct ggml_tensor ** trans_w2;
    struct ggml_tensor ** trans_b2;

    // Transformer weights: final norm
    struct ggml_tensor * trans_final_norm_w;
    struct ggml_tensor * trans_final_norm_b;

    // Transformer weights: LM head (embed_dim x vocab_size)
    struct ggml_tensor * trans_lm_head;
    struct ggml_tensor * trans_lm_head_b;

    // AdamW moment tensors
    struct ggml_tensor ** m_tensors;
    struct ggml_tensor ** v_tensors;

    // AdamW parameters
    float adamw_alpha;
    float adamw_beta1;
    float adamw_beta2;
    float adamw_eps;
    float adamw_wd;
    int adamw_iter;

    // Optimizer params tensor (for ggml_opt_step_adamw)
    struct ggml_tensor * opt_params;

    // Causal attention mask
    struct ggml_tensor * causal_mask;

    // All weight tensors in ordered list
    struct ggml_tensor ** all_weights;
    int num_weight_tensors;
};

static struct compute_ctx compute_begin(ggml_backend_t vk_backend, size_t mem_size, size_t graph_size) {
    struct compute_ctx c;
    struct ggml_init_params params = {
        .mem_size = mem_size,
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    c.ctx = ggml_init(params);
    c.gf = ggml_new_graph_custom(c.ctx, graph_size, true);
    c.buf = NULL;
    return c;
}

static bool compute_alloc(struct compute_ctx * c, ggml_backend_t vk_backend) {
    c->buf = ggml_backend_alloc_ctx_tensors(c->ctx, vk_backend);
    return c->buf != NULL;
}

static bool compute_exec(struct compute_ctx * c, ggml_backend_t vk_backend) {
    enum ggml_status status = ggml_backend_graph_compute(vk_backend, c->gf);
    return status == GGML_STATUS_SUCCESS;
}

static void compute_end(struct compute_ctx * c) {
    if (c->buf) ggml_backend_buffer_free(c->buf);
    if (c->ctx) ggml_free(c->ctx);
}

static void add_weight(struct eve_handle * h, struct ggml_tensor * tensor) {
    h->all_weights[h->num_weight_tensors] = tensor;
    
    // Create moment tensors with same shape
    int n_dims = ggml_n_dims(tensor);
    if (n_dims == 1) {
        h->m_tensors[h->num_weight_tensors] = ggml_new_tensor_1d(h->weight_ctx, GGML_TYPE_F32, tensor->ne[0]);
        h->v_tensors[h->num_weight_tensors] = ggml_new_tensor_1d(h->weight_ctx, GGML_TYPE_F32, tensor->ne[0]);
    } else if (n_dims == 2) {
        h->m_tensors[h->num_weight_tensors] = ggml_new_tensor_2d(h->weight_ctx, GGML_TYPE_F32, tensor->ne[0], tensor->ne[1]);
        h->v_tensors[h->num_weight_tensors] = ggml_new_tensor_2d(h->weight_ctx, GGML_TYPE_F32, tensor->ne[0], tensor->ne[1]);
    } else if (n_dims == 3) {
        h->m_tensors[h->num_weight_tensors] = ggml_new_tensor_3d(h->weight_ctx, GGML_TYPE_F32, tensor->ne[0], tensor->ne[1], tensor->ne[2]);
        h->v_tensors[h->num_weight_tensors] = ggml_new_tensor_3d(h->weight_ctx, GGML_TYPE_F32, tensor->ne[0], tensor->ne[1], tensor->ne[2]);
    } else {
        h->m_tensors[h->num_weight_tensors] = ggml_new_tensor_4d(h->weight_ctx, GGML_TYPE_F32, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
        h->v_tensors[h->num_weight_tensors] = ggml_new_tensor_4d(h->weight_ctx, GGML_TYPE_F32, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    }
    
    h->num_weight_tensors++;
}

eve_handle * eve_create(int vocab_size, int embed_dim, int num_heads, int num_layers,
                        int feed_forward_dim, int max_seq_len) {
    eve_handle * h = (eve_handle *)calloc(1, sizeof(eve_handle));

 // Initialize backends - use CPU for training (Vulkan lacks cross-entropy support)
    h->cpu_backend = ggml_backend_cpu_init();
    h->vk_backend = ggml_backend_vk_init(0);
    if (!h->vk_backend) {
        fprintf(stderr, "Failed to initialize Vulkan backend\n");
        h->vk_backend = h->cpu_backend;
    }
    fprintf(stderr, "Backends: Vulkan=%s, CPU=%s\n",
            ggml_backend_name(h->vk_backend),
            ggml_backend_name(h->cpu_backend));
    h->vk_buft = ggml_backend_get_default_buffer_type(h->vk_backend);

    // Create weight context with no_alloc=true for backend allocation
    size_t weight_mem_size = 2ULL * 1024 * 1024 * 1024; // 2 GB for weights
    struct ggml_init_params params = {
        .mem_size = weight_mem_size,
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    h->weight_ctx = ggml_init(params);
    h->weight_mem_size = weight_mem_size;

    // Store config
    h->vocab_size = vocab_size;
    h->embed_dim = embed_dim;
    h->num_heads = num_heads;
    h->num_layers = num_layers;
    h->feed_forward_dim = feed_forward_dim;
    h->max_seq_len = max_seq_len;

    // Create token embedding
    h->trans_token_emb = ggml_new_tensor_2d(h->weight_ctx, GGML_TYPE_F32, embed_dim, vocab_size);
    ggml_set_param(h->trans_token_emb);

    // Create positional embedding
    h->trans_pos_emb = ggml_new_tensor_2d(h->weight_ctx, GGML_TYPE_F32, embed_dim, max_seq_len);
    ggml_set_param(h->trans_pos_emb);

    // Create per-layer weights
    h->trans_attn_norm_w = (struct ggml_tensor **)calloc(num_layers, sizeof(struct ggml_tensor *));
    h->trans_attn_norm_b = (struct ggml_tensor **)calloc(num_layers, sizeof(struct ggml_tensor *));
    h->trans_wq = (struct ggml_tensor **)calloc(num_layers, sizeof(struct ggml_tensor *));
    h->trans_bq = (struct ggml_tensor **)calloc(num_layers, sizeof(struct ggml_tensor *));
    h->trans_wk = (struct ggml_tensor **)calloc(num_layers, sizeof(struct ggml_tensor *));
    h->trans_bk = (struct ggml_tensor **)calloc(num_layers, sizeof(struct ggml_tensor *));
    h->trans_wv = (struct ggml_tensor **)calloc(num_layers, sizeof(struct ggml_tensor *));
    h->trans_bv = (struct ggml_tensor **)calloc(num_layers, sizeof(struct ggml_tensor *));
    h->trans_wo = (struct ggml_tensor **)calloc(num_layers, sizeof(struct ggml_tensor *));
    h->trans_bo = (struct ggml_tensor **)calloc(num_layers, sizeof(struct ggml_tensor *));
    h->trans_ffn_norm_w = (struct ggml_tensor **)calloc(num_layers, sizeof(struct ggml_tensor *));
    h->trans_ffn_norm_b = (struct ggml_tensor **)calloc(num_layers, sizeof(struct ggml_tensor *));
    h->trans_w1 = (struct ggml_tensor **)calloc(num_layers, sizeof(struct ggml_tensor *));
    h->trans_b1 = (struct ggml_tensor **)calloc(num_layers, sizeof(struct ggml_tensor *));
    h->trans_w2 = (struct ggml_tensor **)calloc(num_layers, sizeof(struct ggml_tensor *));
    h->trans_b2 = (struct ggml_tensor **)calloc(num_layers, sizeof(struct ggml_tensor *));

    for (int i = 0; i < num_layers; i++) {
        h->trans_attn_norm_w[i] = ggml_new_tensor_1d(h->weight_ctx, GGML_TYPE_F32, embed_dim);
        ggml_set_param(h->trans_attn_norm_w[i]);
        h->trans_attn_norm_b[i] = ggml_new_tensor_1d(h->weight_ctx, GGML_TYPE_F32, embed_dim);
        ggml_set_param(h->trans_attn_norm_b[i]);
        h->trans_wq[i] = ggml_new_tensor_2d(h->weight_ctx, GGML_TYPE_F32, embed_dim, embed_dim);
        ggml_set_param(h->trans_wq[i]);
        h->trans_bq[i] = ggml_new_tensor_1d(h->weight_ctx, GGML_TYPE_F32, embed_dim);
        ggml_set_param(h->trans_bq[i]);
        h->trans_wk[i] = ggml_new_tensor_2d(h->weight_ctx, GGML_TYPE_F32, embed_dim, embed_dim);
        ggml_set_param(h->trans_wk[i]);
        h->trans_bk[i] = ggml_new_tensor_1d(h->weight_ctx, GGML_TYPE_F32, embed_dim);
        ggml_set_param(h->trans_bk[i]);
        h->trans_wv[i] = ggml_new_tensor_2d(h->weight_ctx, GGML_TYPE_F32, embed_dim, embed_dim);
        ggml_set_param(h->trans_wv[i]);
        h->trans_bv[i] = ggml_new_tensor_1d(h->weight_ctx, GGML_TYPE_F32, embed_dim);
        ggml_set_param(h->trans_bv[i]);
        h->trans_wo[i] = ggml_new_tensor_2d(h->weight_ctx, GGML_TYPE_F32, embed_dim, embed_dim);
        ggml_set_param(h->trans_wo[i]);
        h->trans_bo[i] = ggml_new_tensor_1d(h->weight_ctx, GGML_TYPE_F32, embed_dim);
        ggml_set_param(h->trans_bo[i]);
        h->trans_ffn_norm_w[i] = ggml_new_tensor_1d(h->weight_ctx, GGML_TYPE_F32, embed_dim);
        ggml_set_param(h->trans_ffn_norm_w[i]);
        h->trans_ffn_norm_b[i] = ggml_new_tensor_1d(h->weight_ctx, GGML_TYPE_F32, embed_dim);
        ggml_set_param(h->trans_ffn_norm_b[i]);
        h->trans_w1[i] = ggml_new_tensor_2d(h->weight_ctx, GGML_TYPE_F32, embed_dim, feed_forward_dim);
        ggml_set_param(h->trans_w1[i]);
        h->trans_b1[i] = ggml_new_tensor_1d(h->weight_ctx, GGML_TYPE_F32, feed_forward_dim);
        ggml_set_param(h->trans_b1[i]);
        h->trans_w2[i] = ggml_new_tensor_2d(h->weight_ctx, GGML_TYPE_F32, feed_forward_dim, embed_dim);
        ggml_set_param(h->trans_w2[i]);
        h->trans_b2[i] = ggml_new_tensor_1d(h->weight_ctx, GGML_TYPE_F32, embed_dim);
        ggml_set_param(h->trans_b2[i]);
    }

    // Create final norm
    h->trans_final_norm_w = ggml_new_tensor_1d(h->weight_ctx, GGML_TYPE_F32, embed_dim);
    ggml_set_param(h->trans_final_norm_w);
    h->trans_final_norm_b = ggml_new_tensor_1d(h->weight_ctx, GGML_TYPE_F32, embed_dim);
    ggml_set_param(h->trans_final_norm_b);

    // Create LM head
    h->trans_lm_head = ggml_new_tensor_2d(h->weight_ctx, GGML_TYPE_F32, embed_dim, vocab_size);
    ggml_set_param(h->trans_lm_head);
    h->trans_lm_head_b = ggml_new_tensor_1d(h->weight_ctx, GGML_TYPE_F32, vocab_size);
    ggml_set_param(h->trans_lm_head_b);

    // Allocate weight tracking arrays
    int num_weights = 2 + num_layers * 16 + 2 + 2; // token_emb, pos_emb, layers, final_norm, lm_head
    h->all_weights = (struct ggml_tensor **)calloc(num_weights, sizeof(struct ggml_tensor *));
    h->m_tensors = (struct ggml_tensor **)calloc(num_weights, sizeof(struct ggml_tensor *));
    h->v_tensors = (struct ggml_tensor **)calloc(num_weights, sizeof(struct ggml_tensor *));
    h->num_weight_tensors = 0;

    // Add all weights to tracking
    add_weight(h, h->trans_token_emb);
    add_weight(h, h->trans_pos_emb);
    for (int i = 0; i < num_layers; i++) {
        add_weight(h, h->trans_attn_norm_w[i]);
        add_weight(h, h->trans_attn_norm_b[i]);
        add_weight(h, h->trans_wq[i]);
        add_weight(h, h->trans_bq[i]);
        add_weight(h, h->trans_wk[i]);
        add_weight(h, h->trans_bk[i]);
        add_weight(h, h->trans_wv[i]);
        add_weight(h, h->trans_bv[i]);
        add_weight(h, h->trans_wo[i]);
        add_weight(h, h->trans_bo[i]);
        add_weight(h, h->trans_ffn_norm_w[i]);
        add_weight(h, h->trans_ffn_norm_b[i]);
        add_weight(h, h->trans_w1[i]);
        add_weight(h, h->trans_b1[i]);
        add_weight(h, h->trans_w2[i]);
        add_weight(h, h->trans_b2[i]);
    }
    add_weight(h, h->trans_final_norm_w);
    add_weight(h, h->trans_final_norm_b);
    add_weight(h, h->trans_lm_head);
    add_weight(h, h->trans_lm_head_b);

    // Initialize AdamW parameters
    h->adamw_alpha = 0.001f;
    h->adamw_beta1 = 0.9f;
    h->adamw_beta2 = 0.999f;
    h->adamw_eps = 1e-8f;
    h->adamw_wd = 0.01f;
    h->adamw_iter = 0;

    // Create optimizer params tensor (7 floats: alpha, beta1, beta2, eps, wd, beta1h, beta2h)
    h->opt_params = ggml_new_tensor_1d(h->weight_ctx, GGML_TYPE_F32, 7);

    // Create causal attention mask (max_seq_len x max_seq_len)
    h->causal_mask = ggml_new_tensor_2d(h->weight_ctx, GGML_TYPE_F32, max_seq_len, max_seq_len);

    // Allocate all tensors (weights + moments + mask) on Vulkan backend for GPU training
    h->weight_buf = ggml_backend_alloc_ctx_tensors(h->weight_ctx, h->vk_backend);
    if (!h->weight_buf) {
        fprintf(stderr, "Failed to allocate weight tensors on Vulkan backend\n");
        eve_destroy(h);
        free(h);
        return NULL;
    }

    // Initialize causal mask: 0 for allowed positions, -inf for future positions
    float * mask_data = (float *)malloc(max_seq_len * max_seq_len * sizeof(float));
    for (int q_pos = 0; q_pos < max_seq_len; q_pos++) {
        for (int k_pos = 0; k_pos < max_seq_len; k_pos++) {
            mask_data[q_pos * max_seq_len + k_pos] = (k_pos > q_pos) ? -INFINITY : 0.0f;
        }
    }
    ggml_backend_tensor_set(h->causal_mask, mask_data, 0, max_seq_len * max_seq_len * sizeof(float));
    free(mask_data);

    // Initialize moment tensors to zero
    for (int i = 0; i < h->num_weight_tensors; i++) {
        size_t m_size = ggml_nbytes(h->m_tensors[i]);
        size_t v_size = ggml_nbytes(h->v_tensors[i]);
        float * zero_data = (float *)calloc(m_size / sizeof(float) + v_size / sizeof(float), sizeof(float));
        ggml_backend_tensor_set(h->m_tensors[i], zero_data, 0, m_size);
        ggml_backend_tensor_set(h->v_tensors[i], zero_data + m_size / sizeof(float), 0, v_size);
        free(zero_data);
    }

    return h;
}

void eve_destroy(eve_handle * h) {
    if (!h) return;

    if (h->weight_ctx) ggml_free(h->weight_ctx);
    if (h->vk_backend) ggml_backend_free(h->vk_backend);
    if (h->cpu_backend) ggml_backend_free(h->cpu_backend);

    free(h->trans_attn_norm_w);
    free(h->trans_attn_norm_b);
    free(h->trans_wq);
    free(h->trans_bq);
    free(h->trans_wk);
    free(h->trans_bk);
    free(h->trans_wv);
    free(h->trans_bv);
    free(h->trans_wo);
    free(h->trans_bo);
    free(h->trans_ffn_norm_w);
    free(h->trans_ffn_norm_b);
    free(h->trans_w1);
    free(h->trans_b1);
    free(h->trans_w2);
    free(h->trans_b2);
    free(h->all_weights);
    free(h->m_tensors);
    free(h->v_tensors);

    free(h);
}

void eve_init_weights(eve_handle * h, unsigned int seed) {
    srand(seed);

    for (int i = 0; i < h->num_weight_tensors; i++) {
        struct ggml_tensor * t = h->all_weights[i];
        int n = ggml_nelements(t);
        
        float scale;
        if (ggml_n_dims(t) >= 2) {
            scale = sqrtf(2.0f / t->ne[1]);
        } else {
            scale = 0.01f;
        }

        float * data = (float *)malloc(n * sizeof(float));
        for (int j = 0; j < n; j++) {
            data[j] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
        }
        ggml_backend_tensor_set(t, data, 0, n * sizeof(float));
        free(data);
    }
}

void eve_set_learning_rate(eve_handle * h, float lr) {
    h->adamw_alpha = lr;
}

// Forward pass through transformer with multi-head self-attention
static struct ggml_tensor * transformer_forward(
    struct ggml_context * ctx,
    struct eve_handle * h,
    struct ggml_tensor * tokens,
    struct ggml_tensor * positions,
    int seq_len)
{
    // Token embedding
    struct ggml_tensor * x = ggml_get_rows(ctx, h->trans_token_emb, tokens);

    // Add positional embedding
    struct ggml_tensor * pos_emb = ggml_get_rows(ctx, h->trans_pos_emb, positions);
    x = ggml_add(ctx, x, pos_emb);

    int head_dim = h->embed_dim / h->num_heads;
    float kq_scale = 1.0f / sqrtf((float)head_dim);

    for (int layer = 0; layer < h->num_layers; layer++) {
        // === Self-attention block ===
        struct ggml_tensor * attn_residual = x;

        // Pre-norm
        x = ggml_rms_norm(ctx, x, 1e-5f);
        x = ggml_mul(ctx, x, h->trans_attn_norm_w[layer]);
        x = ggml_add(ctx, x, h->trans_attn_norm_b[layer]);

        // QKV projections: (embed_dim, seq_len)
        struct ggml_tensor * q = ggml_mul_mat(ctx, h->trans_wq[layer], x);
        q = ggml_add(ctx, q, h->trans_bq[layer]);
        struct ggml_tensor * k = ggml_mul_mat(ctx, h->trans_wk[layer], x);
        k = ggml_add(ctx, k, h->trans_bk[layer]);
        struct ggml_tensor * v = ggml_mul_mat(ctx, h->trans_wv[layer], x);
        v = ggml_add(ctx, v, h->trans_bv[layer]);

        // Reshape to expose heads: (head_dim, num_heads, seq_len)
        q = ggml_reshape_3d(ctx, q, head_dim, h->num_heads, seq_len);
        k = ggml_reshape_3d(ctx, k, head_dim, h->num_heads, seq_len);
        v = ggml_reshape_3d(ctx, v, head_dim, h->num_heads, seq_len);

        // Permute to (head_dim, seq_len, num_heads) so ggml_mul_mat broadcasts over heads
        q = ggml_permute(ctx, q, 0, 2, 1, 3);
        k = ggml_permute(ctx, k, 0, 2, 1, 3);

        // Attention scores: K^T @ Q -> (seq_len, seq_len, num_heads)
        struct ggml_tensor * kq = ggml_mul_mat(ctx, k, q);

        // Causal mask view for current seq_len (must be contiguous for soft_max_ext)
        struct ggml_tensor * mask = ggml_view_2d(ctx, h->causal_mask,
            seq_len, seq_len, h->causal_mask->nb[1], 0);
        mask = ggml_cont(ctx, mask);

        // Fused scale + mask + softmax
        kq = ggml_soft_max_ext(ctx, kq, mask, kq_scale, 0.0f);

        // V handling: permute to (head_dim, seq_len, num_heads), then transpose to (seq_len, head_dim, num_heads)
        // llama.cpp stores V transposed in KV cache; we do it explicitly here
        v = ggml_permute(ctx, v, 0, 2, 1, 3);  // (head_dim, num_heads, seq_len) -> (head_dim, seq_len, num_heads)
        v = ggml_cont(ctx, ggml_transpose(ctx, v));  // -> (seq_len, head_dim, num_heads)

        // kqv = V^T @ softmax(KQ) -> (head_dim, seq_len, num_heads)
        struct ggml_tensor * kqv = ggml_mul_mat(ctx, v, kq);

        // Permute back: (head_dim, seq_len, num_heads) -> (head_dim, num_heads, seq_len)
        kqv = ggml_permute(ctx, kqv, 0, 2, 1, 3);

        // Flatten heads: (head_dim, num_heads, seq_len) -> (embed_dim, seq_len)
        struct ggml_tensor * attn_out = ggml_cont_2d(ctx, kqv, h->embed_dim, seq_len);

        // Output projection
        attn_out = ggml_mul_mat(ctx, h->trans_wo[layer], attn_out);
        attn_out = ggml_add(ctx, attn_out, h->trans_bo[layer]);

        // Residual
        x = ggml_add(ctx, attn_out, attn_residual);

        // === FFN block ===
        struct ggml_tensor * residual = x;
        x = ggml_rms_norm(ctx, x, 1e-5f);
        x = ggml_mul(ctx, x, h->trans_ffn_norm_w[layer]);
        x = ggml_add(ctx, x, h->trans_ffn_norm_b[layer]);

        x = ggml_mul_mat(ctx, h->trans_w1[layer], x);
        x = ggml_add(ctx, x, h->trans_b1[layer]);
        x = ggml_silu(ctx, x);
        x = ggml_mul_mat(ctx, h->trans_w2[layer], x);
        x = ggml_add(ctx, x, h->trans_b2[layer]);

        x = ggml_add(ctx, x, residual);
    }

    // Final norm
    x = ggml_rms_norm(ctx, x, 1e-5f);
    x = ggml_mul(ctx, x, h->trans_final_norm_w);
    x = ggml_add(ctx, x, h->trans_final_norm_b);

    // LM head
    x = ggml_mul_mat(ctx, h->trans_lm_head, x);
    x = ggml_add(ctx, x, h->trans_lm_head_b);

    return x;
}

float eve_train_step(eve_handle * h, int * tokens, int seq_len) {
    if (seq_len < 2) return 0.0f;

    // Create a fresh context for this training step's computation graph
    size_t compute_mem_size = 512ULL * 1024 * 1024; // 512 MB for compute
    struct ggml_init_params params = {
        .mem_size = compute_mem_size,
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) return 0.0f;

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE * 2, true);

    // Input tokens (all but last)
    struct ggml_tensor * input_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq_len - 1);
    struct ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq_len - 1);

    // Forward pass through transformer
    struct ggml_tensor * logits = transformer_forward(ctx, h, input_tokens, positions, seq_len - 1);

    // Decompose cross-entropy into Vulkan-supported ops:
    // CE = -sum(targets * log(softmax(logits)))
    struct ggml_tensor * probs = ggml_soft_max(ctx, logits);
    struct ggml_tensor * log_probs = ggml_log(ctx, probs);

    // Target labels tensor (one-hot encoded)
    struct ggml_tensor * target_labels = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, h->vocab_size, seq_len - 1);
    float * label_data = (float *)malloc(h->vocab_size * (seq_len - 1) * sizeof(float));
    memset(label_data, 0, h->vocab_size * (seq_len - 1) * sizeof(float));
    for (int i = 0; i < seq_len - 1; i++) {
        int target = tokens[i + 1];
        label_data[target * (seq_len - 1) + i] = 1.0f;
    }

    // NLL loss: -sum(targets * log_probs)
    struct ggml_tensor * nll = ggml_mul(ctx, log_probs, target_labels);
    struct ggml_tensor * loss = ggml_sum(ctx, nll);
    loss = ggml_scale(ctx, loss, -1.0f);

    ggml_set_loss(loss);
    ggml_build_forward_expand(gf, loss);
    ggml_build_backward_expand(ctx, gf, NULL);

    // Add optimizer steps to the graph
    h->adamw_iter++;
    float beta1h = 1.0f / (1.0f - powf(h->adamw_beta1, h->adamw_iter));
    float beta2h = 1.0f / (1.0f - powf(h->adamw_beta2, h->adamw_iter));

    // Set optimizer params
    float opt_params_data[7] = {
        h->adamw_alpha,
        h->adamw_beta1,
        h->adamw_beta2,
        h->adamw_eps,
        h->adamw_wd,
        beta1h,
        beta2h
    };
    ggml_backend_tensor_set(h->opt_params, opt_params_data, 0, sizeof(opt_params_data));

    for (int i = 0; i < h->num_weight_tensors; i++) {
        struct ggml_tensor * w = h->all_weights[i];
        struct ggml_tensor * m = h->m_tensors[i];
        struct ggml_tensor * v = h->v_tensors[i];
        struct ggml_tensor * grad = ggml_graph_get_grad(gf, w);

        if (!grad) continue;

        // Add AdamW step to graph (runs on GPU!)
        struct ggml_tensor * opt_step = ggml_opt_step_adamw(ctx, w, grad, m, v, h->opt_params);
        ggml_build_forward_expand(gf, opt_step);
    }

    // Allocate graph tensors on Vulkan backend
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, h->vk_backend);
    if (!buf) {
        free(label_data);
        ggml_free(ctx);
        return 0.0f;
    }

    // Set input data
    for (int i = 0; i < seq_len - 1; i++) {
        ggml_backend_tensor_set(input_tokens, &tokens[i], i * sizeof(int32_t), sizeof(int32_t));
        ggml_backend_tensor_set(positions, &i, i * sizeof(int32_t), sizeof(int32_t));
    }

    // Set target data
    ggml_backend_tensor_set(target_labels, label_data, 0, h->vocab_size * (seq_len - 1) * sizeof(float));
    free(label_data);

    // Reset graph
    ggml_graph_reset(gf);

    // Execute graph on Vulkan backend
    enum ggml_status status = ggml_backend_graph_compute(h->vk_backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return 0.0f;
    }

    // Get loss value
    float loss_val;
    ggml_backend_tensor_get(loss, &loss_val, 0, sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return loss_val;
}

int eve_generate(eve_handle * h, int * prompt_tokens, int prompt_len,
                 int * output_tokens, int max_output_len, float temperature) {
    int generated = 0;
    int * current_tokens = (int *)malloc((prompt_len + max_output_len) * sizeof(int));
    memcpy(current_tokens, prompt_tokens, prompt_len * sizeof(int));

    int start_pos = 0;
    if (prompt_len > h->max_seq_len) {
        start_pos = prompt_len - h->max_seq_len;
    }

    for (int i = 0; i < max_output_len; i++) {
        int seq_len = prompt_len + i - start_pos;
        if (seq_len > h->max_seq_len) seq_len = h->max_seq_len;

        struct compute_ctx c = compute_begin(h->vk_backend, 64 * 1024 * 1024, GGML_DEFAULT_GRAPH_SIZE);

        struct ggml_tensor * input_tokens = ggml_new_tensor_1d(c.ctx, GGML_TYPE_I32, seq_len);
        struct ggml_tensor * positions = ggml_new_tensor_1d(c.ctx, GGML_TYPE_I32, seq_len);

        struct ggml_tensor * logits = transformer_forward(c.ctx, h, input_tokens, positions, seq_len);
        ggml_build_forward_expand(c.gf, logits);

        if (!compute_alloc(&c, h->vk_backend)) {
            compute_end(&c);
            break;
        }

        // Set input data (use tokens from start_pos)
        ggml_backend_tensor_set(input_tokens, &current_tokens[start_pos], 0, seq_len * sizeof(int32_t));
        for (int j = 0; j < seq_len; j++) {
            ggml_backend_tensor_set(positions, &j, j * sizeof(int32_t), sizeof(int32_t));
        }

        if (!compute_exec(&c, h->vk_backend)) {
            compute_end(&c);
            break;
        }

        // Get logits for last position
        int vocab_size = h->vocab_size;
        float * last_logits = (float *)malloc(vocab_size * sizeof(float));
        ggml_backend_tensor_get(logits, last_logits, (seq_len - 1) * vocab_size * sizeof(float), vocab_size * sizeof(float));

        // Apply temperature
        if (temperature > 0.0f && temperature != 1.0f) {
            for (int j = 0; j < vocab_size; j++) {
                last_logits[j] /= temperature;
            }
        }

        // Softmax
        float max_logit = last_logits[0];
        for (int j = 1; j < vocab_size; j++) {
            if (last_logits[j] > max_logit) max_logit = last_logits[j];
        }

        float sum_exp = 0.0f;
        for (int j = 0; j < vocab_size; j++) {
            last_logits[j] = expf(last_logits[j] - max_logit);
            sum_exp += last_logits[j];
        }

        for (int j = 0; j < vocab_size; j++) {
            last_logits[j] /= sum_exp;
        }

        // Sample
        float r = (float)rand() / RAND_MAX;
        float cumsum = 0.0f;
        int sampled_token = 0;
        for (int j = 0; j < vocab_size; j++) {
            cumsum += last_logits[j];
            if (r <= cumsum) {
                sampled_token = j;
                break;
            }
        }

        current_tokens[prompt_len + i] = sampled_token;
        output_tokens[i] = sampled_token;
        generated++;

        free(last_logits);
        compute_end(&c);
    }

    free(current_tokens);
    return generated;
}

int eve_save_model(eve_handle * h, const char * path) {
    FILE * f = fopen(path, "wb");
    if (!f) return -1;

    fwrite(&h->vocab_size, sizeof(int), 1, f);
    fwrite(&h->embed_dim, sizeof(int), 1, f);
    fwrite(&h->num_heads, sizeof(int), 1, f);
    fwrite(&h->num_layers, sizeof(int), 1, f);
    fwrite(&h->feed_forward_dim, sizeof(int), 1, f);
    fwrite(&h->max_seq_len, sizeof(int), 1, f);

    for (int i = 0; i < h->num_weight_tensors; i++) {
        struct ggml_tensor * w = h->all_weights[i];
        int ne0 = (int)w->ne[0];
        int ne1 = (int)w->ne[1];
        fwrite(&ne0, sizeof(int), 1, f);
        fwrite(&ne1, sizeof(int), 1, f);
        size_t nbytes = ggml_nbytes(w);
        float * data = (float *)malloc(nbytes);
        ggml_backend_tensor_get(w, data, 0, nbytes);
        fwrite(data, 1, nbytes, f);
        free(data);
    }

    fclose(f);
    return 0;
}

int eve_load_model(eve_handle * h, const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) return -1;

    int vocab_size, embed_dim, num_heads, num_layers, feed_forward_dim, max_seq_len;
    fread(&vocab_size, sizeof(int), 1, f);
    fread(&embed_dim, sizeof(int), 1, f);
    fread(&num_heads, sizeof(int), 1, f);
    fread(&num_layers, sizeof(int), 1, f);
    fread(&feed_forward_dim, sizeof(int), 1, f);
    fread(&max_seq_len, sizeof(int), 1, f);

    if (vocab_size != h->vocab_size || embed_dim != h->embed_dim ||
        num_heads != h->num_heads || num_layers != h->num_layers ||
        feed_forward_dim != h->feed_forward_dim || max_seq_len != h->max_seq_len) {
        fclose(f);
        return -2;
    }

    for (int i = 0; i < h->num_weight_tensors; i++) {
        struct ggml_tensor * w = h->all_weights[i];
        int ne0, ne1;
        fread(&ne0, sizeof(int), 1, f);
        fread(&ne1, sizeof(int), 1, f);
        size_t nbytes = ggml_nbytes(w);
        float * data = (float *)malloc(nbytes);
        fread(data, 1, nbytes, f);
        ggml_backend_tensor_set(w, data, 0, nbytes);
        free(data);
    }

    fclose(f);
    return 0;
}

int eve_save_checkpoint(eve_handle * h, const char * path) {
    FILE * f = fopen(path, "wb");
    if (!f) return -1;

    // Save config
    fwrite(&h->vocab_size, sizeof(int), 1, f);
    fwrite(&h->embed_dim, sizeof(int), 1, f);
    fwrite(&h->num_heads, sizeof(int), 1, f);
    fwrite(&h->num_layers, sizeof(int), 1, f);
    fwrite(&h->feed_forward_dim, sizeof(int), 1, f);
    fwrite(&h->max_seq_len, sizeof(int), 1, f);

    // Save AdamW iteration count
    fwrite(&h->adamw_iter, sizeof(int), 1, f);

    // Save all weights
    for (int i = 0; i < h->num_weight_tensors; i++) {
        struct ggml_tensor * w = h->all_weights[i];
        int ne0 = (int)w->ne[0];
        int ne1 = (int)w->ne[1];
        fwrite(&ne0, sizeof(int), 1, f);
        fwrite(&ne1, sizeof(int), 1, f);
        size_t nbytes = ggml_nbytes(w);
        float * data = (float *)malloc(nbytes);
        ggml_backend_tensor_get(w, data, 0, nbytes);
        fwrite(data, 1, nbytes, f);
        free(data);
    }

    // Save all moment tensors (m and v)
    for (int i = 0; i < h->num_weight_tensors; i++) {
        struct ggml_tensor * m = h->m_tensors[i];
        struct ggml_tensor * v = h->v_tensors[i];
        size_t m_bytes = ggml_nbytes(m);
        size_t v_bytes = ggml_nbytes(v);
        float * m_data = (float *)malloc(m_bytes);
        float * v_data = (float *)malloc(v_bytes);
        ggml_backend_tensor_get(m, m_data, 0, m_bytes);
        ggml_backend_tensor_get(v, v_data, 0, v_bytes);
        fwrite(m_data, 1, m_bytes, f);
        fwrite(v_data, 1, v_bytes, f);
        free(m_data);
        free(v_data);
    }

    fclose(f);
    return 0;
}

int eve_load_checkpoint(eve_handle * h, const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) return -1;

    // Load and verify config
    int vocab_size, embed_dim, num_heads, num_layers, feed_forward_dim, max_seq_len;
    fread(&vocab_size, sizeof(int), 1, f);
    fread(&embed_dim, sizeof(int), 1, f);
    fread(&num_heads, sizeof(int), 1, f);
    fread(&num_layers, sizeof(int), 1, f);
    fread(&feed_forward_dim, sizeof(int), 1, f);
    fread(&max_seq_len, sizeof(int), 1, f);

    if (vocab_size != h->vocab_size || embed_dim != h->embed_dim ||
        num_heads != h->num_heads || num_layers != h->num_layers ||
        feed_forward_dim != h->feed_forward_dim || max_seq_len != h->max_seq_len) {
        fclose(f);
        return -2;
    }

    // Load AdamW iteration count
    int iter;
    fread(&iter, sizeof(int), 1, f);
    h->adamw_iter = iter;

    // Load all weights
    for (int i = 0; i < h->num_weight_tensors; i++) {
        struct ggml_tensor * w = h->all_weights[i];
        int ne0, ne1;
        fread(&ne0, sizeof(int), 1, f);
        fread(&ne1, sizeof(int), 1, f);
        size_t nbytes = ggml_nbytes(w);
        float * data = (float *)malloc(nbytes);
        fread(data, 1, nbytes, f);
        ggml_backend_tensor_set(w, data, 0, nbytes);
        free(data);
    }

    // Load all moment tensors (m and v)
    for (int i = 0; i < h->num_weight_tensors; i++) {
        struct ggml_tensor * m = h->m_tensors[i];
        struct ggml_tensor * v = h->v_tensors[i];
        size_t m_bytes = ggml_nbytes(m);
        size_t v_bytes = ggml_nbytes(v);
        float * m_data = (float *)malloc(m_bytes);
        float * v_data = (float *)malloc(v_bytes);
        fread(m_data, 1, m_bytes, f);
        fread(v_data, 1, v_bytes, f);
        ggml_backend_tensor_set(m, m_data, 0, m_bytes);
        ggml_backend_tensor_set(v, v_data, 0, v_bytes);
        free(m_data);
        free(v_data);
    }

    fclose(f);
    return iter;
}
