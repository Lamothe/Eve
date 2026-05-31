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

#define NUM_DOWNSAMPLING 4
#define CONV_KERNEL 2
#define CONV_STRIDE 2

#define PROJ_KERNEL 3
#define PROJ_STRIDE 1

#define RES_KERNEL 3
#define RES_STRIDE 1

struct weight_entry {
    char name[128];
    struct ggml_tensor * tensor;
};

struct eve_handle {
    // Vulkan backend
    ggml_backend_t vk_backend;
    ggml_backend_buffer_type_t vk_buft;
    ggml_backend_buffer_t weight_buf;

    // Persistent contexts for weights
    struct ggml_context * weight_ctx;
    size_t weight_mem_size;

    // Weight tensors (encoder)
    struct ggml_tensor * enc_conv[4];
    struct ggml_tensor * enc_conv_bias[4];
    struct ggml_tensor * enc_res_conv1[4];
    struct ggml_tensor * enc_res_bias1[4];
    struct ggml_tensor * enc_res_conv2[4];
    struct ggml_tensor * enc_res_bias2[4];
    struct ggml_tensor * enc_proj;
    struct ggml_tensor * enc_proj_bias;

    // Weight tensors (decoder)
    struct ggml_tensor * dec_proj;
    struct ggml_tensor * dec_proj_bias;
    struct ggml_tensor * dec_tconv[4];
    struct ggml_tensor * dec_tconv_bias[4];
    struct ggml_tensor * dec_res_conv1[4];
    struct ggml_tensor * dec_res_bias1[4];
    struct ggml_tensor * dec_res_conv2[4];
    struct ggml_tensor * dec_res_bias2[4];
    struct ggml_tensor * dec_out;
    struct ggml_tensor * dec_out_bias;

    // Weight tensors (quantizer)
    struct ggml_tensor * codebook;

    // Transformer config
    int embed_dim;
    int num_heads;
    int num_layers;
    int feed_forward_dim;
    int max_seq_len;

    // Transformer weights: token embedding
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

    // Transformer weights: LM head
    struct ggml_tensor * trans_lm_head;
    struct ggml_tensor * trans_lm_head_b;

    // AdamW moment tensors (same shapes as weight tensors)
    struct ggml_tensor ** m_tensors;
    struct ggml_tensor ** v_tensors;

    // AdamW parameters
    float adamw_alpha;
    float adamw_beta1;
    float adamw_beta2;
    float adamw_eps;
    float adamw_wd;

    // VQ-VAE config
    int codebook_size;
    int latent_dim;
    int audio_len;
    float commitment_cost;
    float ema_decay;

    // All weight tensors in ordered list (for iteration)
    struct ggml_tensor ** all_weights;
    int num_weight_tensors;
};

static int conv1d_out_len(int in_len, int kernel, int stride, int padding) {
    return (in_len + 2 * padding - kernel) / stride + 1;
}

static struct ggml_tensor * ggml_conv1d_bias(
    struct ggml_context * ctx,
    struct ggml_tensor * input,
    struct ggml_tensor * weight,
    struct ggml_tensor * bias,
    int stride,
    int padding)
{
    struct ggml_tensor * im2col = ggml_im2col(ctx, weight, input, stride, 0, padding, 0, 1, 0, false, GGML_TYPE_F32);
    struct ggml_tensor * result = ggml_mul_mat(ctx,
        ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1]),
        ggml_reshape_2d(ctx, weight, weight->ne[0] * weight->ne[1], weight->ne[2]));
    result = ggml_reshape_3d(ctx, result, im2col->ne[1], weight->ne[2], im2col->ne[2]);
    struct ggml_tensor * bias_3d = ggml_reshape_3d(ctx, bias, 1, bias->ne[0], 1);
    return ggml_add(ctx, result, bias_3d);
}

static struct ggml_tensor * ggml_residual_block(
    struct ggml_context * ctx,
    struct ggml_tensor * input,
    int channels,
    struct ggml_tensor * conv1_w, struct ggml_tensor * conv1_b,
    struct ggml_tensor * conv2_w, struct ggml_tensor * conv2_b)
{
    struct ggml_tensor * h = ggml_conv1d_bias(ctx, input, conv1_w, conv1_b, 1, 1);
    h = ggml_silu(ctx, h);
    h = ggml_conv1d_bias(ctx, h, conv2_w, conv2_b, 1, 1);
    return ggml_add(ctx, h, input);
}

static struct ggml_tensor * ggml_upsample_conv_bias(
    struct ggml_context * ctx,
    struct ggml_tensor * input,
    struct ggml_tensor * weight,
    struct ggml_tensor * bias)
{
    struct ggml_tensor * reshaped = ggml_reshape_4d(ctx, input, input->ne[0], 1, input->ne[1], input->ne[2]);
    int64_t ne[] = { reshaped->ne[0], 2, reshaped->ne[2], reshaped->ne[3] };
    struct ggml_tensor * target = ggml_new_tensor_4d(ctx, reshaped->type, ne[0], ne[1], ne[2], ne[3]);
    struct ggml_tensor * up = ggml_repeat(ctx, reshaped, target);
    struct ggml_tensor * up_3d = ggml_reshape_3d(ctx, up, up->ne[0] * up->ne[1], up->ne[2], up->ne[3]);
    return ggml_conv1d_bias(ctx, up_3d, weight, bias, 1, 1);
}

// ============================================================
// Helpers: compute graph with Vulkan backend
// ============================================================
// Builds a graph in a no_alloc context, allocates GPU memory,
// sets inputs, computes on GPU, and reads outputs.

struct compute_ctx {
    struct ggml_context * ctx;
    struct ggml_cgraph * gf;
    ggml_backend_buffer_t buf;
};

static struct compute_ctx compute_begin(ggml_backend_t vk_backend, size_t meta_mem, int graph_size) {
    struct compute_ctx c;
    struct ggml_init_params params = {
        .mem_size   = meta_mem,
        .mem_buffer = NULL,
        .no_alloc   = true,
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

#define TENSOR_OVERHEAD() ggml_tensor_overhead()
#define GRAPH_OVERHEAD(n) ggml_graph_overhead_custom(n, false)

// ============================================================
// API: create
// ============================================================

static size_t estimate_weight_mem(int codebook_size, int latent_dim) {
    size_t total = 0;
    total += 4 * 4 * 1 * 2;
    total += 4 * 4 * 2 * 4;
    total += 4 * 4 * 4 * 8;
    total += 4 * 4 * 8 * 16;
    total += 4 * 4 * 16 * 64;
    for (int c = 2; c <= 16; c *= 2)
        total += 2 * 3 * c * c;
    total += 4 * 4 * 64 * 16;
    for (int c = 16; c >= 2; c /= 2)
        total += 4 * 4 * c * (c/2);
    for (int c = 16; c >= 2; c /= 2)
        total += 2 * 3 * c * c;
    total += 4 * 4 * 2 * 1;
    total += codebook_size * latent_dim;
    total *= 4;
    total *= 2;
    total += 1024 * 1024 * 100;
    return total;
}

eve_handle* eve_create(
    int codebook_size,
    int latent_dim,
    int embed_dim,
    int num_heads,
    int num_layers,
    int feed_forward_dim,
    int max_seq_len,
    int sample_rate,
    int audio_len)
{
    (void)sample_rate;

    eve_handle * h = (eve_handle *)calloc(1, sizeof(eve_handle));
    if (!h) return NULL;

    h->vk_backend = NULL;
    h->vk_buft = NULL;
    h->weight_buf = NULL;

    h->codebook_size = codebook_size;
    h->latent_dim = latent_dim;
    h->audio_len = audio_len;

    h->adamw_alpha = 1e-5f;
    h->adamw_beta1 = 0.9f;
    h->adamw_beta2 = 0.999f;
    h->adamw_eps   = 1e-8f;
    h->adamw_wd    = 0.0f;

    h->commitment_cost = 1.0f;
    h->ema_decay = 0.99f;

    // Initialize Vulkan backend
    h->vk_backend = ggml_backend_vk_init(0);
    if (!h->vk_backend) {
        fprintf(stderr, "eve: failed to initialize Vulkan backend\n");
        free(h);
        return NULL;
    }
    h->vk_buft = ggml_backend_vk_buffer_type(0);

    // Estimate memory for weight context (metadata only, data allocated from GPU buffer)
    size_t mem = estimate_weight_mem(codebook_size, latent_dim);
    h->weight_mem_size = mem;

    struct ggml_init_params params = {
        .mem_size   = mem,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };

    h->weight_ctx = ggml_init(params);
    if (!h->weight_ctx) {
        ggml_backend_free(h->vk_backend);
        free(h);
        return NULL;
    }

    int channels = 1;
    for (int i = 0; i < NUM_DOWNSAMPLING; i++) {
        int out_ch = channels * 2;
        h->enc_conv[i] = ggml_new_tensor_3d(h->weight_ctx, GGML_TYPE_F32, CONV_KERNEL, channels, out_ch);
        h->enc_conv_bias[i] = ggml_new_tensor_1d(h->weight_ctx, GGML_TYPE_F32, out_ch);
        h->enc_res_conv1[i] = ggml_new_tensor_3d(h->weight_ctx, GGML_TYPE_F32, RES_KERNEL, out_ch, out_ch);
        h->enc_res_bias1[i] = ggml_new_tensor_1d(h->weight_ctx, GGML_TYPE_F32, out_ch);
        h->enc_res_conv2[i] = ggml_new_tensor_3d(h->weight_ctx, GGML_TYPE_F32, RES_KERNEL, out_ch, out_ch);
        h->enc_res_bias2[i] = ggml_new_tensor_1d(h->weight_ctx, GGML_TYPE_F32, out_ch);
        channels = out_ch;
    }

    h->enc_proj = ggml_new_tensor_3d(h->weight_ctx, GGML_TYPE_F32, PROJ_KERNEL, channels, latent_dim);
    h->enc_proj_bias = ggml_new_tensor_1d(h->weight_ctx, GGML_TYPE_F32, latent_dim);

    int dec_start = channels;
    h->dec_proj = ggml_new_tensor_3d(h->weight_ctx, GGML_TYPE_F32, PROJ_KERNEL, latent_dim, dec_start);
    h->dec_proj_bias = ggml_new_tensor_1d(h->weight_ctx, GGML_TYPE_F32, dec_start);

    for (int i = 0; i < NUM_DOWNSAMPLING; i++) {
        int in_ch = dec_start;
        int out_ch = dec_start / 2;
        h->dec_tconv[i] = ggml_new_tensor_3d(h->weight_ctx, GGML_TYPE_F32, 3, in_ch, out_ch);
        h->dec_tconv_bias[i] = ggml_new_tensor_1d(h->weight_ctx, GGML_TYPE_F32, out_ch);
        h->dec_res_conv1[i] = ggml_new_tensor_3d(h->weight_ctx, GGML_TYPE_F32, RES_KERNEL, out_ch, out_ch);
        h->dec_res_bias1[i] = ggml_new_tensor_1d(h->weight_ctx, GGML_TYPE_F32, out_ch);
        h->dec_res_conv2[i] = ggml_new_tensor_3d(h->weight_ctx, GGML_TYPE_F32, RES_KERNEL, out_ch, out_ch);
        h->dec_res_bias2[i] = ggml_new_tensor_1d(h->weight_ctx, GGML_TYPE_F32, out_ch);
        dec_start = out_ch;
    }

    h->dec_out = ggml_new_tensor_3d(h->weight_ctx, GGML_TYPE_F32, PROJ_KERNEL, dec_start, 1);
    h->dec_out_bias = ggml_new_tensor_1d(h->weight_ctx, GGML_TYPE_F32, 1);

    h->codebook = ggml_new_tensor_2d(h->weight_ctx, GGML_TYPE_F32, latent_dim, codebook_size);

    // Store Transformer config
    h->embed_dim = embed_dim;
    h->num_heads = num_heads;
    h->num_layers = num_layers;
    h->feed_forward_dim = feed_forward_dim;
    h->max_seq_len = max_seq_len;

    // Allocate Transformer weight arrays
    h->trans_token_emb = ggml_new_tensor_2d(h->weight_ctx, GGML_TYPE_F32, embed_dim, codebook_size);
    h->trans_pos_emb = ggml_new_tensor_2d(h->weight_ctx, GGML_TYPE_F32, embed_dim, max_seq_len);

    int n = num_layers;
    h->trans_attn_norm_w = (struct ggml_tensor **)calloc(n, sizeof(struct ggml_tensor *));
    h->trans_attn_norm_b = (struct ggml_tensor **)calloc(n, sizeof(struct ggml_tensor *));
    h->trans_wq = (struct ggml_tensor **)calloc(n, sizeof(struct ggml_tensor *));
    h->trans_bq = (struct ggml_tensor **)calloc(n, sizeof(struct ggml_tensor *));
    h->trans_wk = (struct ggml_tensor **)calloc(n, sizeof(struct ggml_tensor *));
    h->trans_bk = (struct ggml_tensor **)calloc(n, sizeof(struct ggml_tensor *));
    h->trans_wv = (struct ggml_tensor **)calloc(n, sizeof(struct ggml_tensor *));
    h->trans_bv = (struct ggml_tensor **)calloc(n, sizeof(struct ggml_tensor *));
    h->trans_wo = (struct ggml_tensor **)calloc(n, sizeof(struct ggml_tensor *));
    h->trans_bo = (struct ggml_tensor **)calloc(n, sizeof(struct ggml_tensor *));
    h->trans_ffn_norm_w = (struct ggml_tensor **)calloc(n, sizeof(struct ggml_tensor *));
    h->trans_ffn_norm_b = (struct ggml_tensor **)calloc(n, sizeof(struct ggml_tensor *));
    h->trans_w1 = (struct ggml_tensor **)calloc(n, sizeof(struct ggml_tensor *));
    h->trans_b1 = (struct ggml_tensor **)calloc(n, sizeof(struct ggml_tensor *));
    h->trans_w2 = (struct ggml_tensor **)calloc(n, sizeof(struct ggml_tensor *));
    h->trans_b2 = (struct ggml_tensor **)calloc(n, sizeof(struct ggml_tensor *));

    h->trans_final_norm_w = ggml_new_tensor_1d(h->weight_ctx, GGML_TYPE_F32, embed_dim);
    h->trans_final_norm_b = ggml_new_tensor_1d(h->weight_ctx, GGML_TYPE_F32, embed_dim);
    h->trans_lm_head = ggml_new_tensor_2d(h->weight_ctx, GGML_TYPE_F32, embed_dim, codebook_size);
    h->trans_lm_head_b = ggml_new_tensor_1d(h->weight_ctx, GGML_TYPE_F32, codebook_size);

    for (int i = 0; i < n; i++) {
        h->trans_attn_norm_w[i] = ggml_new_tensor_1d(h->weight_ctx, GGML_TYPE_F32, embed_dim);
        h->trans_attn_norm_b[i] = ggml_new_tensor_1d(h->weight_ctx, GGML_TYPE_F32, embed_dim);
        h->trans_wq[i] = ggml_new_tensor_2d(h->weight_ctx, GGML_TYPE_F32, embed_dim, embed_dim);
        h->trans_bq[i] = ggml_new_tensor_1d(h->weight_ctx, GGML_TYPE_F32, embed_dim);
        h->trans_wk[i] = ggml_new_tensor_2d(h->weight_ctx, GGML_TYPE_F32, embed_dim, embed_dim);
        h->trans_bk[i] = ggml_new_tensor_1d(h->weight_ctx, GGML_TYPE_F32, embed_dim);
        h->trans_wv[i] = ggml_new_tensor_2d(h->weight_ctx, GGML_TYPE_F32, embed_dim, embed_dim);
        h->trans_bv[i] = ggml_new_tensor_1d(h->weight_ctx, GGML_TYPE_F32, embed_dim);
        h->trans_wo[i] = ggml_new_tensor_2d(h->weight_ctx, GGML_TYPE_F32, embed_dim, embed_dim);
        h->trans_bo[i] = ggml_new_tensor_1d(h->weight_ctx, GGML_TYPE_F32, embed_dim);
        h->trans_ffn_norm_w[i] = ggml_new_tensor_1d(h->weight_ctx, GGML_TYPE_F32, embed_dim);
        h->trans_ffn_norm_b[i] = ggml_new_tensor_1d(h->weight_ctx, GGML_TYPE_F32, embed_dim);
        h->trans_w1[i] = ggml_new_tensor_2d(h->weight_ctx, GGML_TYPE_F32, embed_dim, feed_forward_dim);
        h->trans_b1[i] = ggml_new_tensor_1d(h->weight_ctx, GGML_TYPE_F32, feed_forward_dim);
        h->trans_w2[i] = ggml_new_tensor_2d(h->weight_ctx, GGML_TYPE_F32, feed_forward_dim, embed_dim);
        h->trans_b2[i] = ggml_new_tensor_1d(h->weight_ctx, GGML_TYPE_F32, embed_dim);
    }

    // Count Transformer weights
    int trans_wt_count = 6 + n * 16; // 2 embed + n*16 layer + 2 final_norm + 2 lm_head

    h->num_weight_tensors = 0;
    h->num_weight_tensors += 4 * 6 + 2;  // encoder
    h->num_weight_tensors += 2 + 4 * 6 + 2;  // decoder
    h->num_weight_tensors += 1;  // codebook
    h->num_weight_tensors += 6 + num_layers * 16;  // transformer

    h->all_weights = (struct ggml_tensor **)calloc(h->num_weight_tensors, sizeof(struct ggml_tensor *));
    h->m_tensors = (struct ggml_tensor **)calloc(h->num_weight_tensors, sizeof(struct ggml_tensor *));
    h->v_tensors = (struct ggml_tensor **)calloc(h->num_weight_tensors, sizeof(struct ggml_tensor *));

    int wi = 0;
    auto add_weight = [&](struct ggml_tensor * t) {
        h->all_weights[wi] = t;
        h->m_tensors[wi] = ggml_new_tensor_3d(h->weight_ctx, t->type,
            t->ne[0], t->ne[1], t->ne[2]);
        h->v_tensors[wi] = ggml_new_tensor_3d(h->weight_ctx, t->type,
            t->ne[0], t->ne[1], t->ne[2]);
        ggml_set_param(t);
        wi++;
    };

    for (int i = 0; i < NUM_DOWNSAMPLING; i++) {
        add_weight(h->enc_conv[i]);
        add_weight(h->enc_conv_bias[i]);
        add_weight(h->enc_res_conv1[i]);
        add_weight(h->enc_res_bias1[i]);
        add_weight(h->enc_res_conv2[i]);
        add_weight(h->enc_res_bias2[i]);
    }
    add_weight(h->enc_proj);
    add_weight(h->enc_proj_bias);

    add_weight(h->dec_proj);
    add_weight(h->dec_proj_bias);
    for (int i = 0; i < NUM_DOWNSAMPLING; i++) {
        add_weight(h->dec_tconv[i]);
        add_weight(h->dec_tconv_bias[i]);
        add_weight(h->dec_res_conv1[i]);
        add_weight(h->dec_res_bias1[i]);
        add_weight(h->dec_res_conv2[i]);
        add_weight(h->dec_res_bias2[i]);
    }
    add_weight(h->dec_out);
    add_weight(h->dec_out_bias);
    add_weight(h->codebook);

    // Add Transformer weights
    add_weight(h->trans_token_emb);
    add_weight(h->trans_pos_emb);
    for (int i = 0; i < n; i++) {
        add_weight(h->trans_attn_norm_w[i]);
        add_weight(h->trans_attn_norm_b[i]);
        add_weight(h->trans_wq[i]);
        add_weight(h->trans_bq[i]);
        add_weight(h->trans_wk[i]);
        add_weight(h->trans_bk[i]);
        add_weight(h->trans_wv[i]);
        add_weight(h->trans_bv[i]);
        add_weight(h->trans_wo[i]);
        add_weight(h->trans_bo[i]);
        add_weight(h->trans_ffn_norm_w[i]);
        add_weight(h->trans_ffn_norm_b[i]);
        add_weight(h->trans_w1[i]);
        add_weight(h->trans_b1[i]);
        add_weight(h->trans_w2[i]);
        add_weight(h->trans_b2[i]);
    }
    add_weight(h->trans_final_norm_w);
    add_weight(h->trans_final_norm_b);
    add_weight(h->trans_lm_head);
    add_weight(h->trans_lm_head_b);

    if (wi != h->num_weight_tensors) {
        fprintf(stderr, "eve: weight count mismatch: %d vs %d\n", wi, h->num_weight_tensors);
    }

    // Allocate all weight/moment tensor data from GPU buffer
    h->weight_buf = ggml_backend_alloc_ctx_tensors_from_buft(h->weight_ctx, h->vk_buft);
    if (!h->weight_buf) {
        fprintf(stderr, "eve: failed to allocate GPU weight buffer\n");
        ggml_free(h->weight_ctx);
        ggml_backend_free(h->vk_backend);
        free(h->all_weights);
        free(h->m_tensors);
        free(h->v_tensors);
        free(h);
        return NULL;
    }
    ggml_backend_buffer_set_usage(h->weight_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // Initialize m, v to zero on GPU
    for (int wi = 0; wi < h->num_weight_tensors; wi++) {
        int nbytes = (int)ggml_nbytes(h->m_tensors[wi]);
        float * zeros = (float *)calloc(1, nbytes);
        ggml_backend_tensor_set(h->m_tensors[wi], zeros, 0, nbytes);
        ggml_backend_tensor_set(h->v_tensors[wi], zeros, 0, nbytes);
        free(zeros);
    }

    // Initialize biases to zero on GPU
    auto zero_bias = [&](struct ggml_tensor * t) {
        int nbytes = (int)ggml_nbytes(t);
        float * zeros = (float *)calloc(1, nbytes);
        ggml_backend_tensor_set(t, zeros, 0, nbytes);
        free(zeros);
    };
    for (int i = 0; i < NUM_DOWNSAMPLING; i++) {
        zero_bias(h->enc_conv_bias[i]);
        zero_bias(h->enc_res_bias1[i]);
        zero_bias(h->enc_res_bias2[i]);
    }
    zero_bias(h->enc_proj_bias);
    zero_bias(h->dec_proj_bias);
    for (int i = 0; i < NUM_DOWNSAMPLING; i++) {
        zero_bias(h->dec_tconv_bias[i]);
        zero_bias(h->dec_res_bias1[i]);
        zero_bias(h->dec_res_bias2[i]);
    }
    zero_bias(h->dec_out_bias);

    return h;
}

void eve_destroy(eve_handle * h) {
    if (!h) return;
    if (h->weight_buf) ggml_backend_buffer_free(h->weight_buf);
    if (h->weight_ctx) ggml_free(h->weight_ctx);
    if (h->vk_backend) ggml_backend_free(h->vk_backend);
    // Free Transformer weight arrays
    if (h->trans_attn_norm_w) free(h->trans_attn_norm_w);
    if (h->trans_attn_norm_b) free(h->trans_attn_norm_b);
    if (h->trans_wq) free(h->trans_wq);
    if (h->trans_bq) free(h->trans_bq);
    if (h->trans_wk) free(h->trans_wk);
    if (h->trans_bk) free(h->trans_bk);
    if (h->trans_wv) free(h->trans_wv);
    if (h->trans_bv) free(h->trans_bv);
    if (h->trans_wo) free(h->trans_wo);
    if (h->trans_bo) free(h->trans_bo);
    if (h->trans_ffn_norm_w) free(h->trans_ffn_norm_w);
    if (h->trans_ffn_norm_b) free(h->trans_ffn_norm_b);
    if (h->trans_w1) free(h->trans_w1);
    if (h->trans_b1) free(h->trans_b1);
    if (h->trans_w2) free(h->trans_w2);
    if (h->trans_b2) free(h->trans_b2);
    free(h->all_weights);
    free(h->m_tensors);
    free(h->v_tensors);
    free(h);
}

// ============================================================
// API: init weights
// ============================================================

void eve_init_weights(eve_handle * h, unsigned int seed) {
    srand(seed);

    auto init_tensor = [&](struct ggml_tensor * t) {
        int n = (int)ggml_nelements(t);
        float scale = 1.0f / sqrtf((float)n);
        float * cpu = (float *)malloc(n * sizeof(float));
        for (int i = 0; i < n; i++)
            cpu[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
        ggml_backend_tensor_set(t, cpu, 0, n * sizeof(float));
        free(cpu);
    };

    auto init_bias = [&](struct ggml_tensor * t) {
        int nbytes = (int)ggml_nbytes(t);
        float * zeros = (float *)calloc(1, nbytes);
        ggml_backend_tensor_set(t, zeros, 0, nbytes);
        free(zeros);
    };

    for (int i = 0; i < NUM_DOWNSAMPLING; i++) {
        init_tensor(h->enc_conv[i]);
        init_bias(h->enc_conv_bias[i]);
        init_tensor(h->enc_res_conv1[i]);
        init_bias(h->enc_res_bias1[i]);
        init_tensor(h->enc_res_conv2[i]);
        init_bias(h->enc_res_bias2[i]);
    }
    init_tensor(h->enc_proj);
    init_bias(h->enc_proj_bias);

    init_tensor(h->dec_proj);
    init_bias(h->dec_proj_bias);
    for (int i = 0; i < NUM_DOWNSAMPLING; i++) {
        init_tensor(h->dec_tconv[i]);
        init_bias(h->dec_tconv_bias[i]);
        init_tensor(h->dec_res_conv1[i]);
        init_bias(h->dec_res_bias1[i]);
        init_tensor(h->dec_res_conv2[i]);
        init_bias(h->dec_res_bias2[i]);
    }
    init_tensor(h->dec_out);
    init_bias(h->dec_out_bias);

    init_tensor(h->codebook);

    // Init Transformer weights
    init_tensor(h->trans_token_emb);
    init_tensor(h->trans_pos_emb);
    for (int i = 0; i < h->num_layers; i++) {
        init_tensor(h->trans_attn_norm_w[i]);
        init_bias(h->trans_attn_norm_b[i]);
        init_tensor(h->trans_wq[i]);
        init_bias(h->trans_bq[i]);
        init_tensor(h->trans_wk[i]);
        init_bias(h->trans_bk[i]);
        init_tensor(h->trans_wv[i]);
        init_bias(h->trans_bv[i]);
        init_tensor(h->trans_wo[i]);
        init_bias(h->trans_bo[i]);
        init_tensor(h->trans_ffn_norm_w[i]);
        init_bias(h->trans_ffn_norm_b[i]);
        init_tensor(h->trans_w1[i]);
        init_bias(h->trans_b1[i]);
        init_tensor(h->trans_w2[i]);
        init_bias(h->trans_b2[i]);
    }
    init_tensor(h->trans_final_norm_w);
    init_bias(h->trans_final_norm_b);
    init_tensor(h->trans_lm_head);
    init_bias(h->trans_lm_head_b);
}

// ============================================================
// API: save/load weights
// ============================================================

int eve_save_weights(eve_handle * h, const char * path) {
    int n = 0;
    n += 4 * 6 + 2;
    n += 2 + 4 * 6 + 2;
    n += 1;
    n += 6 + h->num_layers * 16;  // Transformer weights

    FILE * f = fopen(path, "wb");
    if (!f) return -1;

    auto save_tensor = [&](const char * name, struct ggml_tensor * t) {
        int name_len = (int)strlen(name) + 1;
        int nbytes = (int)ggml_nbytes(t);
        float * cpu = (float *)malloc(nbytes);
        ggml_backend_tensor_get(t, cpu, 0, nbytes);
        fwrite(&name_len, 4, 1, f);
        fwrite(name, 1, name_len, f);
        fwrite(&nbytes, 4, 1, f);
        fwrite(cpu, 1, nbytes, f);
        free(cpu);
    };

    fwrite(&n, 4, 1, f);

    char name[128];
    for (int i = 0; i < NUM_DOWNSAMPLING; i++) {
        snprintf(name, sizeof(name), "encoder/conv%d/weight", i); save_tensor(name, h->enc_conv[i]);
        snprintf(name, sizeof(name), "encoder/conv%d/bias", i); save_tensor(name, h->enc_conv_bias[i]);
        snprintf(name, sizeof(name), "encoder/res%d/conv1/weight", i); save_tensor(name, h->enc_res_conv1[i]);
        snprintf(name, sizeof(name), "encoder/res%d/conv1/bias", i); save_tensor(name, h->enc_res_bias1[i]);
        snprintf(name, sizeof(name), "encoder/res%d/conv2/weight", i); save_tensor(name, h->enc_res_conv2[i]);
        snprintf(name, sizeof(name), "encoder/res%d/conv2/bias", i); save_tensor(name, h->enc_res_bias2[i]);
    }
    save_tensor("encoder/proj/weight", h->enc_proj);
    save_tensor("encoder/proj/bias", h->enc_proj_bias);

    save_tensor("decoder/proj/weight", h->dec_proj);
    save_tensor("decoder/proj/bias", h->dec_proj_bias);
    for (int i = 0; i < NUM_DOWNSAMPLING; i++) {
        snprintf(name, sizeof(name), "decoder/tconv%d/weight", i); save_tensor(name, h->dec_tconv[i]);
        snprintf(name, sizeof(name), "decoder/tconv%d/bias", i); save_tensor(name, h->dec_tconv_bias[i]);
        snprintf(name, sizeof(name), "decoder/res%d/conv1/weight", i); save_tensor(name, h->dec_res_conv1[i]);
        snprintf(name, sizeof(name), "decoder/res%d/conv1/bias", i); save_tensor(name, h->dec_res_bias1[i]);
        snprintf(name, sizeof(name), "decoder/res%d/conv2/weight", i); save_tensor(name, h->dec_res_conv2[i]);
        snprintf(name, sizeof(name), "decoder/res%d/conv2/bias", i); save_tensor(name, h->dec_res_bias2[i]);
    }
    save_tensor("decoder/out/weight", h->dec_out);
    save_tensor("decoder/out/bias", h->dec_out_bias);

    save_tensor("quantizer/codebook", h->codebook);

    // Save Transformer weights
    save_tensor("transformer/token_emb", h->trans_token_emb);
    save_tensor("transformer/pos_emb", h->trans_pos_emb);
    for (int i = 0; i < h->num_layers; i++) {
        snprintf(name, sizeof(name), "transformer/layer%d/attn_norm/w", i); save_tensor(name, h->trans_attn_norm_w[i]);
        snprintf(name, sizeof(name), "transformer/layer%d/attn_norm/b", i); save_tensor(name, h->trans_attn_norm_b[i]);
        snprintf(name, sizeof(name), "transformer/layer%d/q/w", i); save_tensor(name, h->trans_wq[i]);
        snprintf(name, sizeof(name), "transformer/layer%d/q/b", i); save_tensor(name, h->trans_bq[i]);
        snprintf(name, sizeof(name), "transformer/layer%d/k/w", i); save_tensor(name, h->trans_wk[i]);
        snprintf(name, sizeof(name), "transformer/layer%d/k/b", i); save_tensor(name, h->trans_bk[i]);
        snprintf(name, sizeof(name), "transformer/layer%d/v/w", i); save_tensor(name, h->trans_wv[i]);
        snprintf(name, sizeof(name), "transformer/layer%d/v/b", i); save_tensor(name, h->trans_bv[i]);
        snprintf(name, sizeof(name), "transformer/layer%d/wo/w", i); save_tensor(name, h->trans_wo[i]);
        snprintf(name, sizeof(name), "transformer/layer%d/wo/b", i); save_tensor(name, h->trans_bo[i]);
        snprintf(name, sizeof(name), "transformer/layer%d/ffn_norm/w", i); save_tensor(name, h->trans_ffn_norm_w[i]);
        snprintf(name, sizeof(name), "transformer/layer%d/ffn_norm/b", i); save_tensor(name, h->trans_ffn_norm_b[i]);
        snprintf(name, sizeof(name), "transformer/layer%d/w1", i); save_tensor(name, h->trans_w1[i]);
        snprintf(name, sizeof(name), "transformer/layer%d/b1", i); save_tensor(name, h->trans_b1[i]);
        snprintf(name, sizeof(name), "transformer/layer%d/w2", i); save_tensor(name, h->trans_w2[i]);
        snprintf(name, sizeof(name), "transformer/layer%d/b2", i); save_tensor(name, h->trans_b2[i]);
    }
    save_tensor("transformer/final_norm/w", h->trans_final_norm_w);
    save_tensor("transformer/final_norm/b", h->trans_final_norm_b);
    save_tensor("transformer/lm_head", h->trans_lm_head);
    save_tensor("transformer/lm_head_b", h->trans_lm_head_b);

    fclose(f);
    return 0;
}

int eve_load_weights(eve_handle * h, const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) return -1;

    auto load_tensor = [&](const char * expected_name, struct ggml_tensor * t) {
        int name_len;
        fread(&name_len, 4, 1, f);
        char * name = (char *)malloc(name_len);
        fread(name, 1, name_len, f);
        int nbytes;
        fread(&nbytes, 4, 1, f);
        if ((int)ggml_nbytes(t) != nbytes) {
            fprintf(stderr, "eve: weight size mismatch for '%s': expected %d, got %d\n",
                    expected_name, (int)ggml_nbytes(t), nbytes);
            free(name);
            return -1;
        }
        float * cpu = (float *)malloc(nbytes);
        fread(cpu, 1, nbytes, f);
        ggml_backend_tensor_set(t, cpu, 0, nbytes);
        free(cpu);
        free(name);
        return 0;
    };

    int n;
    fread(&n, 4, 1, f);

    for (int i = 0; i < NUM_DOWNSAMPLING; i++) {
        load_tensor("encoder/conv/weight", h->enc_conv[i]);
        load_tensor("encoder/conv/bias", h->enc_conv_bias[i]);
        load_tensor("encoder/res/conv1/weight", h->enc_res_conv1[i]);
        load_tensor("encoder/res/conv1/bias", h->enc_res_bias1[i]);
        load_tensor("encoder/res/conv2/weight", h->enc_res_conv2[i]);
        load_tensor("encoder/res/conv2/bias", h->enc_res_bias2[i]);
    }
    load_tensor("encoder/proj/weight", h->enc_proj);
    load_tensor("encoder/proj/bias", h->enc_proj_bias);

    load_tensor("decoder/proj/weight", h->dec_proj);
    load_tensor("decoder/proj/bias", h->dec_proj_bias);
    for (int i = 0; i < NUM_DOWNSAMPLING; i++) {
        load_tensor("decoder/tconv/weight", h->dec_tconv[i]);
        load_tensor("decoder/tconv/bias", h->dec_tconv_bias[i]);
        load_tensor("decoder/res/conv1/weight", h->dec_res_conv1[i]);
        load_tensor("decoder/res/conv1/bias", h->dec_res_bias1[i]);
        load_tensor("decoder/res/conv2/weight", h->dec_res_conv2[i]);
        load_tensor("decoder/res/conv2/bias", h->dec_res_bias2[i]);
    }
    load_tensor("decoder/out/weight", h->dec_out);
    load_tensor("decoder/out/bias", h->dec_out_bias);

    load_tensor("quantizer/codebook", h->codebook);

    // Load Transformer weights
    load_tensor("transformer/token_emb", h->trans_token_emb);
    load_tensor("transformer/pos_emb", h->trans_pos_emb);
    for (int i = 0; i < h->num_layers; i++) {
        load_tensor("transformer/layer0/attn_norm/w", h->trans_attn_norm_w[i]);
        load_tensor("transformer/layer0/attn_norm/b", h->trans_attn_norm_b[i]);
        load_tensor("transformer/layer0/q/w", h->trans_wq[i]);
        load_tensor("transformer/layer0/q/b", h->trans_bq[i]);
        load_tensor("transformer/layer0/k/w", h->trans_wk[i]);
        load_tensor("transformer/layer0/k/b", h->trans_bk[i]);
        load_tensor("transformer/layer0/v/w", h->trans_wv[i]);
        load_tensor("transformer/layer0/v/b", h->trans_bv[i]);
        load_tensor("transformer/layer0/wo/w", h->trans_wo[i]);
        load_tensor("transformer/layer0/wo/b", h->trans_bo[i]);
        load_tensor("transformer/layer0/ffn_norm/w", h->trans_ffn_norm_w[i]);
        load_tensor("transformer/layer0/ffn_norm/b", h->trans_ffn_norm_b[i]);
        load_tensor("transformer/layer0/w1", h->trans_w1[i]);
        load_tensor("transformer/layer0/b1", h->trans_b1[i]);
        load_tensor("transformer/layer0/w2", h->trans_w2[i]);
        load_tensor("transformer/layer0/b2", h->trans_b2[i]);
    }
    load_tensor("transformer/final_norm/w", h->trans_final_norm_w);
    load_tensor("transformer/final_norm/b", h->trans_final_norm_b);
    load_tensor("transformer/lm_head", h->trans_lm_head);
    load_tensor("transformer/lm_head_b", h->trans_lm_head_b);

    fclose(f);
    return 0;
}

// ============================================================
// API: VQ-VAE encoder forward
// ============================================================

int eve_encoder_forward(eve_handle * h, const float * audio, int batch, int audio_len,
                        float * z, int z_capacity)
{
    size_t meta_mem = TENSOR_OVERHEAD() * 1000 + GRAPH_OVERHEAD(GGML_DEFAULT_GRAPH_SIZE);
    struct compute_ctx c = compute_begin(h->vk_backend, meta_mem, GGML_DEFAULT_GRAPH_SIZE);

    struct ggml_tensor * inp = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F32, audio_len, 1, batch);
    struct ggml_tensor * cur = inp;
    int seq_len = audio_len;

    for (int i = 0; i < NUM_DOWNSAMPLING; i++) {
        int out_ch = (1 << (i + 1));
        cur = ggml_conv1d_bias(c.ctx, cur, h->enc_conv[i], h->enc_conv_bias[i], CONV_STRIDE, 0);
        seq_len = conv1d_out_len(seq_len, CONV_KERNEL, CONV_STRIDE, 0);
        cur = ggml_residual_block(c.ctx, cur, out_ch,
                                  h->enc_res_conv1[i], h->enc_res_bias1[i],
                                  h->enc_res_conv2[i], h->enc_res_bias2[i]);
    }

    cur = ggml_conv1d_bias(c.ctx, cur, h->enc_proj, h->enc_proj_bias, PROJ_STRIDE, 1);

    ggml_set_output(cur);
    ggml_build_forward_expand(c.gf, cur);

    if (!compute_alloc(&c, h->vk_backend)) {
        compute_end(&c);
        return -1;
    }

    ggml_backend_tensor_set(inp, audio, 0, batch * audio_len * sizeof(float));

    if (!compute_exec(&c, h->vk_backend)) {
        compute_end(&c);
        return -1;
    }

    int out_len = seq_len;
    int needed = batch * h->latent_dim * out_len;
    if (needed > z_capacity) {
        compute_end(&c);
        return -1;
    }

    ggml_backend_tensor_get(cur, z, 0, needed * sizeof(float));
    compute_end(&c);
    return out_len;
}

// ============================================================
// API: quantize (CPU-based - manual argmin)
// ============================================================

void eve_quantize(eve_handle * h, const float * z, int batch, int seq_len,
                  int * codes, float * quantized, int z_capacity)
{
    // Read codebook from GPU to CPU for manual distance computation
    int cb_size = h->codebook_size;
    int ld = h->latent_dim;
    int nbytes = ggml_nbytes(h->codebook);
    float * cb_cpu = (float *)malloc(nbytes);
    ggml_backend_tensor_get(h->codebook, cb_cpu, 0, nbytes);

    int n = seq_len * batch;
    for (int i = 0; i < n; i++) {
        float best_dist = INFINITY;
        int best_idx = 0;
        for (int j = 0; j < cb_size; j++) {
            float dist = 0.0f;
            for (int k = 0; k < ld; k++) {
                float diff = z[i * ld + k] - cb_cpu[j * ld + k];
                dist += diff * diff;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = j;
            }
        }
        codes[i] = best_idx;
        for (int k = 0; k < ld; k++)
            quantized[i * ld + k] = cb_cpu[best_idx * ld + k];
    }
    free(cb_cpu);
}

// ============================================================
// API: encode (encoder + quantize)
// ============================================================

int eve_encode(eve_handle * h, const float * audio, int batch, int audio_len,
               int * codes, int codes_capacity)
{
    int seq_len = batch * h->latent_dim * (audio_len / 16 + 1);
    float * z = (float *)malloc(seq_len * sizeof(float));

    int out_len = eve_encoder_forward(h, audio, batch, audio_len, z, seq_len);
    if (out_len < 0) {
        free(z);
        return -1;
    }

    if (batch * out_len > codes_capacity) {
        free(z);
        return -1;
    }

    float * quantized = (float *)malloc(batch * out_len * h->latent_dim * sizeof(float));
    eve_quantize(h, z, batch, out_len, codes, quantized, batch * out_len * h->latent_dim);
    free(quantized);
    free(z);

    return out_len;
}

// ============================================================
// API: set learning rate
// ============================================================

void eve_set_learning_rate(eve_handle * h, float lr) {
    h->adamw_alpha = lr;
}

void eve_set_commitment_cost(eve_handle * h, float cost) {
    h->commitment_cost = cost;
}

void eve_set_ema_decay(eve_handle * h, float decay) {
    h->ema_decay = decay;
}

// ============================================================
// Training graph builders
// ============================================================

static struct ggml_tensor * build_encoder_graph(
    struct ggml_context * ctx,
    struct ggml_tensor * audio_input,
    eve_handle * h,
    int audio_len)
{
    (void)audio_len;
    struct ggml_tensor * cur = audio_input;
    for (int i = 0; i < NUM_DOWNSAMPLING; i++) {
        cur = ggml_conv1d_bias(ctx, cur, h->enc_conv[i], h->enc_conv_bias[i], CONV_STRIDE, 0);
        cur = ggml_residual_block(ctx, cur, (1 << (i + 1)),
                                  h->enc_res_conv1[i], h->enc_res_bias1[i],
                                  h->enc_res_conv2[i], h->enc_res_bias2[i]);
    }
    cur = ggml_conv1d_bias(ctx, cur, h->enc_proj, h->enc_proj_bias, PROJ_STRIDE, 1);
    return cur;
}

static struct ggml_tensor * build_decoder_graph(
    struct ggml_context * ctx,
    struct ggml_tensor * z_input,
    eve_handle * h,
    int z_len)
{
    (void)z_len;
    struct ggml_tensor * cur = ggml_conv1d_bias(ctx, z_input, h->dec_proj, h->dec_proj_bias, PROJ_STRIDE, 1);
    for (int i = 0; i < NUM_DOWNSAMPLING; i++) {
        int out_ch = (1 << (NUM_DOWNSAMPLING - i - 1));
        cur = ggml_upsample_conv_bias(ctx, cur, h->dec_tconv[i], h->dec_tconv_bias[i]);
        cur = ggml_residual_block(ctx, cur, out_ch,
                                  h->dec_res_conv1[i], h->dec_res_bias1[i],
                                  h->dec_res_conv2[i], h->dec_res_bias2[i]);
    }
    cur = ggml_conv1d_bias(ctx, cur, h->dec_out, h->dec_out_bias, PROJ_STRIDE, 1);
    return cur;
}

// ============================================================
// Helper: create AdamW params tensor
// ============================================================

// ============================================================
// Helper: add AdamW ops to a backward graph
// ============================================================
// Returns the adamw_params tensor (data must be set after GPU alloc).

static struct ggml_tensor * add_adamw_ops(struct ggml_context * ctx, struct ggml_cgraph * gf, eve_handle * h) {
    struct ggml_tensor * adamw_p = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 7);
    for (int wi = 0; wi < h->num_weight_tensors; wi++) {
        struct ggml_tensor * grad = ggml_graph_get_grad(gf, h->all_weights[wi]);
        if (grad) {
            struct ggml_tensor * opt = ggml_opt_step_adamw(ctx,
                h->all_weights[wi], grad, h->m_tensors[wi], h->v_tensors[wi], adamw_p);
            ggml_set_output(opt);
        }
    }
    return adamw_p;
}

static void set_adamw_params(struct ggml_tensor * p, eve_handle * h) {
    float pdata[7] = {
        h->adamw_alpha, h->adamw_beta1, h->adamw_beta2,
        h->adamw_eps, h->adamw_wd,
        h->adamw_beta1 * h->adamw_beta1,
        h->adamw_beta2 * h->adamw_beta2,
    };
    ggml_backend_tensor_set(p, pdata, 0, 7 * sizeof(float));
}

// ============================================================
// VQ-VAE training step
// ============================================================

float eve_train_vqvae_step(eve_handle * h, const float * audio, int batch, int audio_len) {
    int z_len = audio_len / 16;
    int z_elements = batch * h->latent_dim * z_len;

    // ---- 1. Encoder forward (separate graph, no backward) ----
    float * z = (float *)calloc(z_elements, sizeof(float));
    int computed_len = eve_encoder_forward(h, audio, batch, audio_len, z, z_elements);
    if (computed_len != z_len) {
        free(z);
        return -1.0f;
    }

    // ---- 2. Manual argmin + codebook lookup (CPU) ----
    int * codes = (int *)calloc(batch * z_len, sizeof(int));
    float * qz = (float *)calloc(z_elements, sizeof(float));
    eve_quantize(h, z, batch, z_len, codes, qz, z_elements);

    // ---- 3. EMA codebook update ----
    {
        int cb_size = h->codebook_size;
        int ld = h->latent_dim;
        float decay = h->ema_decay;

        // Read codebook from GPU
        int cb_nbytes = (int)ggml_nbytes(h->codebook);
        float * cb_cpu = (float *)malloc(cb_nbytes);
        ggml_backend_tensor_get(h->codebook, cb_cpu, 0, cb_nbytes);

        float * usage = (float *)calloc(cb_size, sizeof(float));
        float * accum = (float *)calloc(cb_size * ld, sizeof(float));
        for (int i = 0; i < batch * z_len; i++) {
            int c = codes[i];
            usage[c] += 1.0f;
            for (int k = 0; k < ld; k++)
                accum[c * ld + k] += qz[i * ld + k];
        }
        for (int j = 0; j < cb_size; j++) {
            if (usage[j] > 0) {
                float inv = 1.0f / usage[j];
                for (int k = 0; k < ld; k++) {
                    float v = decay * cb_cpu[j * ld + k] + (1.0f - decay) * accum[j * ld + k] * inv;
                    if (v > 5.0f) v = 5.0f;
                    if (v < -5.0f) v = -5.0f;
                    cb_cpu[j * ld + k] = v;
                }
            }
        }

        // Write updated codebook back to GPU
        ggml_backend_tensor_set(h->codebook, cb_cpu, 0, cb_nbytes);

        free(usage);
        free(accum);
        free(cb_cpu);
    }

    // ---- 4. Decoder backward (GPU) ----
    float recon_loss = 0.0f;
    float * decoder_grad_at_input = (float *)calloc(z_elements, sizeof(float));
    {
        // Build decoder graph with backward + AdamW on GPU
        size_t meta_mem = TENSOR_OVERHEAD() * 2000 + GRAPH_OVERHEAD(GGML_DEFAULT_GRAPH_SIZE * 4);
        struct compute_ctx c = compute_begin(h->vk_backend, meta_mem, GGML_DEFAULT_GRAPH_SIZE * 4);

        struct ggml_tensor * inp = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F32, z_len, h->latent_dim, batch);
        struct ggml_tensor * recon = build_decoder_graph(c.ctx, inp, h, z_len);
        struct ggml_tensor * target = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F32, audio_len, 1, batch);
        struct ggml_tensor * loss = ggml_mean(c.ctx, ggml_sqr(c.ctx, ggml_sub(c.ctx, recon, target)));
        ggml_set_loss(loss);

        ggml_build_forward_expand(c.gf, loss);
        ggml_build_backward_expand(c.ctx, c.gf, NULL);

        // Add AdamW for all weights
        struct ggml_tensor * adamw_p = add_adamw_ops(c.ctx, c.gf, h);

        if (!compute_alloc(&c, h->vk_backend)) {
            compute_end(&c);
            free(z); free(codes); free(qz); free(decoder_grad_at_input);
            return -1.0f;
        }

        set_adamw_params(adamw_p, h);

        // Set input data
        ggml_backend_tensor_set(inp, qz, 0, z_elements * sizeof(float));
        ggml_backend_tensor_set(target, audio, 0, batch * audio_len * sizeof(float));

        if (!compute_exec(&c, h->vk_backend)) {
            compute_end(&c);
            free(z); free(codes); free(qz); free(decoder_grad_at_input);
            return -1.0f;
        }

        // Read loss
        ggml_backend_tensor_get(loss, &recon_loss, 0, sizeof(float));

        // Read gradient at decoder input
        struct ggml_tensor * inp_grad = ggml_graph_get_grad(c.gf, inp);
        if (inp_grad) {
            ggml_backend_tensor_get(inp_grad, decoder_grad_at_input, 0, z_elements * sizeof(float));
        }

        compute_end(&c);
    }

    // ---- 5. Encoder backward (GPU) ----
    float commit_loss = 0.0f;
    {
        // Compute combined gradient for z
        float * grad_z = (float *)calloc(z_elements, sizeof(float));
        float commit_scale = 2.0f * h->commitment_cost / z_elements;

        // Read codebook from GPU for commit loss calculation
        int cb_nbytes = (int)ggml_nbytes(h->codebook);
        float * cb_cpu = (float *)malloc(cb_nbytes);
        ggml_backend_tensor_get(h->codebook, cb_cpu, 0, cb_nbytes);

        int ld = h->latent_dim;
        float sum_sq = 0;
        for (int i = 0; i < batch * z_len; i++) {
            int c = codes[i];
            for (int k = 0; k < ld; k++) {
                float diff = z[i * ld + k] - cb_cpu[c * ld + k];
                sum_sq += diff * diff;
                grad_z[i * ld + k] = commit_scale * diff + decoder_grad_at_input[i * ld + k];
            }
        }
        commit_loss = h->commitment_cost * sum_sq / z_elements;
        free(cb_cpu);

        // Build encoder graph with backward + AdamW on GPU
        size_t meta_mem = TENSOR_OVERHEAD() * 2000 + GRAPH_OVERHEAD(GGML_DEFAULT_GRAPH_SIZE * 4);
        struct compute_ctx c = compute_begin(h->vk_backend, meta_mem, GGML_DEFAULT_GRAPH_SIZE * 4);

        struct ggml_tensor * inp = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F32, audio_len, 1, batch);
        struct ggml_tensor * z_out = build_encoder_graph(c.ctx, inp, h, audio_len);

        // Fake loss with custom gradient
        struct ggml_tensor * grad_t = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F32, z_len, h->latent_dim, batch);
        struct ggml_tensor * fake_loss = ggml_sum(c.ctx, ggml_mul(c.ctx, z_out, grad_t));
        ggml_set_loss(fake_loss);

        ggml_build_forward_expand(c.gf, fake_loss);
        ggml_build_backward_expand(c.ctx, c.gf, NULL);

        // Add AdamW for encoder weights only
        struct ggml_tensor * adamw_p = add_adamw_ops(c.ctx, c.gf, h);

        if (!compute_alloc(&c, h->vk_backend)) {
            compute_end(&c);
            free(z); free(codes); free(qz); free(decoder_grad_at_input); free(grad_z);
            return -1.0f;
        }

        set_adamw_params(adamw_p, h);

        // Set input data
        ggml_backend_tensor_set(inp, audio, 0, batch * audio_len * sizeof(float));
        ggml_backend_tensor_set(grad_t, grad_z, 0, z_elements * sizeof(float));

        if (!compute_exec(&c, h->vk_backend)) {
            compute_end(&c);
            free(z); free(codes); free(qz); free(decoder_grad_at_input); free(grad_z);
            return -1.0f;
        }

        compute_end(&c);
        free(grad_z);
    }

    free(z);
    free(codes);
    free(qz);
    free(decoder_grad_at_input);

    return recon_loss + commit_loss;
}

float eve_train_transformer_step(eve_handle * h, const int * codes, int batch, int seq_len) {
    int embed_dim = h->embed_dim;
    int num_heads = h->num_heads;
    int d_k = embed_dim / num_heads;
    int ff_dim = h->feed_forward_dim;
    int codebook_size = h->codebook_size;
    int num_layers = h->num_layers;
    int out_len = seq_len - 1;
    if (out_len <= 0) return -1.0f;
    int total_tokens = seq_len * batch;
    int total_targets = out_len * batch;

    // Shift targets on CPU
    int * tgt_cpu = (int *)malloc(total_targets * sizeof(int));
    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < out_len; s++) {
            tgt_cpu[s * batch + b] = codes[(s + 1) * batch + b];
        }
    }

    // Build compute context
    struct compute_ctx c = compute_begin(h->vk_backend,
        TENSOR_OVERHEAD() * 10000 + GRAPH_OVERHEAD(GGML_DEFAULT_GRAPH_SIZE * 4),
        GGML_DEFAULT_GRAPH_SIZE * 4);

    struct ggml_cgraph * gf = ggml_new_graph_custom(c.ctx, GGML_DEFAULT_GRAPH_SIZE * 4, true);

    // Create inputs in c.ctx directly!
    struct ggml_tensor * tokens = ggml_new_tensor_1d(c.ctx, GGML_TYPE_I32, total_tokens);
    struct ggml_tensor * pos_ids = ggml_new_tensor_1d(c.ctx, GGML_TYPE_I32, total_tokens);
    struct ggml_tensor * targets = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, codebook_size, total_targets);

    // One-hot encode targets on CPU
    float * tgt_one_hot = (float *)calloc(codebook_size * total_targets, sizeof(float));
    for (int i = 0; i < total_targets; i++) {
        int target_idx = tgt_cpu[i];
        if (target_idx >= 0 && target_idx < codebook_size) {
            tgt_one_hot[i * codebook_size + target_idx] = 1.0f;
        }
    }

    // Allocate CPU buffers for input data
    int * tokens_cpu = (int *)malloc(total_tokens * sizeof(int));
    int * pos_cpu = (int *)malloc(total_tokens * sizeof(int));

    // Fill input data
    for (int i = 0; i < total_tokens; i++)
        tokens_cpu[i] = codes[i];
    for (int i = 0; i < total_tokens; i++)
        pos_cpu[i] = i % seq_len;

    // Token embedding lookup: tokens [N] -> [N, embed_dim]
    // token_emb shape: [embed_dim, codebook_size]
    struct ggml_tensor * tok_emb = ggml_get_rows(c.ctx, h->trans_token_emb, tokens);

    // Positional embedding lookup: pos_ids [N] -> [N, embed_dim]
    // pos_emb shape: [embed_dim, max_seq_len]
    struct ggml_tensor * pos_emb = ggml_get_rows(c.ctx, h->trans_pos_emb, pos_ids);

    // Add embeddings: [N, embed_dim]
    struct ggml_tensor * x = ggml_add(c.ctx, tok_emb, pos_emb);

    // Reshape to [embed_dim, seq_len, batch] for layer processing
    x = ggml_reshape_3d(c.ctx, x, embed_dim, seq_len, batch);

    // Transformer layers
    for (int i = 0; i < num_layers; i++) {
        // --- Multi-head self-attention ---
        struct ggml_tensor * norm1 = ggml_rms_norm(c.ctx, x, 1e-5f);
        norm1 = ggml_add(c.ctx, ggml_mul(c.ctx, norm1, h->trans_attn_norm_w[i]), h->trans_attn_norm_b[i]);

        // Q, K, V projections: [embed_dim, seq_len, batch]
        struct ggml_tensor * q = ggml_add(c.ctx, ggml_mul_mat(c.ctx, h->trans_wq[i], norm1), h->trans_bq[i]);
        struct ggml_tensor * k = ggml_add(c.ctx, ggml_mul_mat(c.ctx, h->trans_wk[i], norm1), h->trans_bk[i]);
        struct ggml_tensor * v = ggml_add(c.ctx, ggml_mul_mat(c.ctx, h->trans_wv[i], norm1), h->trans_bv[i]);

        // Reshape for multi-head
        q = ggml_reshape_4d(c.ctx, q, d_k, num_heads, seq_len, batch);
        k = ggml_reshape_4d(c.ctx, k, d_k, num_heads, seq_len, batch);
        v = ggml_reshape_4d(c.ctx, v, d_k, num_heads, seq_len, batch);

        printf("Layer %d Q shape: [%ld, %ld, %ld, %ld]\n", i, q->ne[0], q->ne[1], q->ne[2], q->ne[3]);
        printf("Layer %d K shape: [%ld, %ld, %ld, %ld]\n", i, k->ne[0], k->ne[1], k->ne[2], k->ne[3]);
        printf("Layer %d V shape: [%ld, %ld, %ld, %ld]\n", i, v->ne[0], v->ne[1], v->ne[2], v->ne[3]);

        // Permute to:
        // Q: [d_k, seq_len, num_heads, batch]
        // K: [d_k, seq_len, num_heads, batch]
        // V: [seq_len, d_k, num_heads, batch]
        struct ggml_tensor * q_perm = ggml_permute(c.ctx, q, 0, 2, 1, 3);
        struct ggml_tensor * k_perm = ggml_permute(c.ctx, k, 0, 2, 1, 3);
        struct ggml_tensor * v_perm = ggml_permute(c.ctx, v, 1, 2, 0, 3);

        printf("Layer %d q_perm shape: [%ld, %ld, %ld, %ld]\n", i, q_perm->ne[0], q_perm->ne[1], q_perm->ne[2], q_perm->ne[3]);
        printf("Layer %d k_perm shape: [%ld, %ld, %ld, %ld]\n", i, k_perm->ne[0], k_perm->ne[1], k_perm->ne[2], k_perm->ne[3]);
        printf("Layer %d v_perm shape: [%ld, %ld, %ld, %ld]\n", i, v_perm->ne[0], v_perm->ne[1], v_perm->ne[2], v_perm->ne[3]);

        // Attention scores: Q * K^T / sqrt(d_k)
        // kq = ggml_mul_mat(k_perm, q_perm) -> [seq_len, seq_len, num_heads, batch]
        struct ggml_tensor * kq = ggml_mul_mat(c.ctx, ggml_cont(c.ctx, k_perm), q_perm);
        float scale = 1.0f / sqrtf((float)d_k);
        kq = ggml_scale(c.ctx, kq, scale);

        // Causal mask
        kq = ggml_diag_mask_inf(c.ctx, kq, 0);

        // Softmax
        struct ggml_tensor * attn = ggml_soft_max(c.ctx, kq);

        printf("Layer %d attn shape: [%ld, %ld, %ld, %ld]\n", i, attn->ne[0], attn->ne[1], attn->ne[2], attn->ne[3]);

        // Attention output: attn * V
        // attn_out = ggml_mul_mat(v_perm, attn) -> [d_k, seq_len, num_heads, batch]
        struct ggml_tensor * attn_out = ggml_mul_mat(c.ctx, ggml_cont(c.ctx, v_perm), attn);

        // Permute back to: [d_k, num_heads, seq_len, batch]
        attn_out = ggml_permute(c.ctx, attn_out, 0, 2, 1, 3);

        // Reshape to: [embed_dim, seq_len, batch]
        attn_out = ggml_reshape_3d(c.ctx, ggml_cont(c.ctx, attn_out), embed_dim, seq_len, batch);

        // Output projection
        struct ggml_tensor * attn_proj = ggml_add(c.ctx, ggml_mul_mat(c.ctx, h->trans_wo[i], attn_out), h->trans_bo[i]);

        // Residual connection
        x = ggml_add(c.ctx, x, attn_proj);

        // --- Feed-forward network ---
        struct ggml_tensor * norm2 = ggml_rms_norm(c.ctx, x, 1e-5f);
        norm2 = ggml_add(c.ctx, ggml_mul(c.ctx, norm2, h->trans_ffn_norm_w[i]), h->trans_ffn_norm_b[i]);

        // Up projection + SiLU activation
        struct ggml_tensor * ffn_up = ggml_mul_mat(c.ctx, h->trans_w1[i], norm2);
        ffn_up = ggml_add(c.ctx, ffn_up, h->trans_b1[i]);
        ffn_up = ggml_silu(c.ctx, ffn_up);

        // Down projection
        struct ggml_tensor * ffn_out = ggml_add(c.ctx, ggml_mul_mat(c.ctx, h->trans_w2[i], ffn_up), h->trans_b2[i]);

        // Residual connection
        x = ggml_add(c.ctx, x, ffn_out);
    }

    // Final LayerNorm
    x = ggml_rms_norm(c.ctx, x, 1e-5f);
    x = ggml_add(c.ctx, ggml_mul(c.ctx, x, h->trans_final_norm_w), h->trans_final_norm_b);

    // LM head: [embed_dim, seq_len, batch] -> [codebook_size, seq_len, batch]
    struct ggml_tensor * logits = ggml_add(c.ctx, ggml_mul_mat(c.ctx, h->trans_lm_head, x), h->trans_lm_head_b);

    // Slice logits to seq_len - 1: [codebook_size, out_len, batch] (using ggml_view_3d)
    logits = ggml_view_3d(c.ctx, logits, codebook_size, out_len, batch, logits->nb[1], logits->nb[2], 0);

    // Reshape logits to 2D: [codebook_size, out_len * batch]
    struct ggml_tensor * logits_2d = ggml_reshape_2d(c.ctx, logits, codebook_size, total_targets);
    ggml_set_output(logits_2d);

    // Build forward and backward graphs separately
    struct ggml_cgraph * gf_forward = ggml_new_graph_custom(c.ctx, GGML_DEFAULT_GRAPH_SIZE * 2, false);
    ggml_build_forward_expand(gf_forward, logits_2d);

    struct ggml_cgraph * gf_backward = ggml_new_graph_custom(c.ctx, GGML_DEFAULT_GRAPH_SIZE * 4, true);
    ggml_build_forward_expand(gf_backward, logits_2d);

    // Create dummy loss tensor (actual cross-entropy computed on CPU)
    struct ggml_tensor * loss = ggml_sum(c.ctx, logits_2d);
    ggml_set_loss(loss);
    ggml_build_forward_expand(gf_backward, loss);
    ggml_build_backward_expand(c.ctx, gf_backward, NULL);
    struct ggml_tensor * adamw_p = add_adamw_ops(c.ctx, gf_backward, h);

    if (!compute_alloc(&c, h->vk_backend)) {
        free(tokens_cpu);
        free(pos_cpu);
        free(tgt_cpu);
        free(tgt_one_hot);
        return -1.0f;
    }

    // Set input data
    ggml_backend_tensor_set(tokens, tokens_cpu, 0, total_tokens * sizeof(int));
    ggml_backend_tensor_set(pos_ids, pos_cpu, 0, total_tokens * sizeof(int));

    // Run forward pass on GPU
    ggml_backend_graph_compute(h->vk_backend, gf_forward);

    // Read computed logits from GPU -> CPU
    float * logits_cpu = (float *)malloc(codebook_size * total_targets * sizeof(float));
    ggml_backend_tensor_get(logits_2d, logits_cpu, 0, codebook_size * total_targets * sizeof(float));

    // Compute cross-entropy loss and gradients on CPU
    float * grad_logits_cpu = (float *)calloc(codebook_size * total_targets, sizeof(float));
    float loss_val = 0.0f;

    for (int col = 0; col < total_targets; col++) {
        // Stability offset
        float max_logit = -INFINITY;
        for (int i = 0; i < codebook_size; i++) {
            float val = logits_cpu[col * codebook_size + i];
            if (val > max_logit) max_logit = val;
        }

        float sum_exp = 0.0f;
        for (int i = 0; i < codebook_size; i++) {
            sum_exp += expf(logits_cpu[col * codebook_size + i] - max_logit);
        }

        for (int i = 0; i < codebook_size; i++) {
            float prob = expf(logits_cpu[col * codebook_size + i] - max_logit) / sum_exp;
            float target_val = tgt_one_hot[col * codebook_size + i];

            if (target_val > 0.0f) {
                loss_val -= logf(prob + 1e-15f);
            }

            // Gradient: (prob - target) / total_targets
            grad_logits_cpu[col * codebook_size + i] = (prob - target_val) / total_targets;
        }
    }
    loss_val /= total_targets;

    // Write computed gradients to GPU logits->grad tensor
    struct ggml_tensor * logits_grad = ggml_graph_get_grad(gf_backward, logits_2d);
    if (logits_grad) {
        ggml_backend_tensor_set(logits_grad, grad_logits_cpu, 0, codebook_size * total_targets * sizeof(float));
    }

    set_adamw_params(adamw_p, h);

    // Run backward pass and optimizer updates on GPU
    ggml_backend_graph_compute(h->vk_backend, gf_backward);

    compute_end(&c);
    free(tokens_cpu);
    free(pos_cpu);
    free(tgt_cpu);
    free(tgt_one_hot);
    free(logits_cpu);
    free(grad_logits_cpu);

    return loss_val;

    return loss_val;
}

static int sample_token_temperature(const float * logits, int codebook_size, float temp) {
    float max_logit = -INFINITY;
    for (int i = 0; i < codebook_size; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    float * probs = (float *)malloc(codebook_size * sizeof(float));
    float sum_exp = 0.0f;
    for (int i = 0; i < codebook_size; i++) {
        probs[i] = expf((logits[i] - max_logit) / temp);
        sum_exp += probs[i];
    }
    for (int i = 0; i < codebook_size; i++) {
        probs[i] /= sum_exp;
    }

    float r = (float)rand() / (float)RAND_MAX;
    float cum = 0.0f;
    for (int i = 0; i < codebook_size; i++) {
        cum += probs[i];
        if (r <= cum) { free(probs); return i; }
    }
    free(probs);
    return codebook_size - 1;
}

void eve_generate(eve_handle * h, const float * prompt_audio, int prompt_len,
                  int num_tokens, float temperature, float * output_audio, int output_capacity)
{
    srand(12345);
    int codebook_size = h->codebook_size;
    int latent_dim = h->latent_dim;
    int embed_dim = h->embed_dim;
    int num_heads = h->num_heads;
    int d_k = embed_dim / num_heads;
    int num_layers = h->num_layers;

    // Step 1: Encode prompt audio to codes
    int prompt_z_len = prompt_len / 16;
    float * z = (float *)calloc(prompt_z_len * latent_dim, sizeof(float));
    int encoded_z_len = eve_encoder_forward(h, prompt_audio, 1, prompt_len, z, prompt_z_len * latent_dim);
    if (encoded_z_len <= 0) { free(z); return; }

    int * codes = (int *)calloc((encoded_z_len + num_tokens), sizeof(int));
    float * qz = (float *)calloc(encoded_z_len * latent_dim, sizeof(float));
    eve_quantize(h, z, 1, encoded_z_len, codes, qz, encoded_z_len * latent_dim);
    int current_len = encoded_z_len;

    free(z);
    free(qz);

    // Step 2: Autoregressive generation (one token at a time)
    float * logits_buf = (float *)malloc(codebook_size * sizeof(float));

    for (int gen = 0; gen < num_tokens; gen++) {
        int seq_len = current_len + 1;

        struct compute_ctx c = compute_begin(h->vk_backend,
            TENSOR_OVERHEAD() * 10000 + GRAPH_OVERHEAD(GGML_DEFAULT_GRAPH_SIZE * 4),
            GGML_DEFAULT_GRAPH_SIZE * 4);

        struct ggml_cgraph * gf = ggml_new_graph_custom(c.ctx, GGML_DEFAULT_GRAPH_SIZE * 4, false);

        struct ggml_tensor * tokens = ggml_new_tensor_1d(c.ctx, GGML_TYPE_I32, seq_len);
        struct ggml_tensor * pos_ids = ggml_new_tensor_1d(c.ctx, GGML_TYPE_I32, seq_len);

        int * tokens_cpu = (int *)malloc(seq_len * sizeof(int));
        int * pos_cpu = (int *)malloc(seq_len * sizeof(int));
        for (int i = 0; i < seq_len; i++) {
            tokens_cpu[i] = codes[i];
            pos_cpu[i] = i;
        }

        struct ggml_tensor * tok_emb = ggml_get_rows(c.ctx, h->trans_token_emb, tokens);
        struct ggml_tensor * pos_emb = ggml_get_rows(c.ctx, h->trans_pos_emb, pos_ids);
        struct ggml_tensor * x = ggml_add(c.ctx, tok_emb, pos_emb);
        x = ggml_reshape_3d(c.ctx, x, embed_dim, seq_len, 1);

        for (int i = 0; i < num_layers; i++) {
            struct ggml_tensor * norm1 = ggml_rms_norm(c.ctx, x, 1e-5f);
            norm1 = ggml_add(c.ctx, ggml_mul(c.ctx, norm1, h->trans_attn_norm_w[i]), h->trans_attn_norm_b[i]);

            struct ggml_tensor * q = ggml_add(c.ctx, ggml_mul_mat(c.ctx, h->trans_wq[i], norm1), h->trans_bq[i]);
            struct ggml_tensor * k = ggml_add(c.ctx, ggml_mul_mat(c.ctx, h->trans_wk[i], norm1), h->trans_bk[i]);
            struct ggml_tensor * v = ggml_add(c.ctx, ggml_mul_mat(c.ctx, h->trans_wv[i], norm1), h->trans_bv[i]);

            q = ggml_reshape_4d(c.ctx, q, d_k, num_heads, seq_len, 1);
            k = ggml_reshape_4d(c.ctx, k, d_k, num_heads, seq_len, 1);
            v = ggml_reshape_4d(c.ctx, v, d_k, num_heads, seq_len, 1);

            struct ggml_tensor * q_perm = ggml_permute(c.ctx, q, 0, 2, 1, 3);
            struct ggml_tensor * k_perm = ggml_permute(c.ctx, k, 0, 2, 1, 3);
            struct ggml_tensor * v_perm = ggml_permute(c.ctx, v, 1, 2, 0, 3);

            struct ggml_tensor * kq = ggml_mul_mat(c.ctx, ggml_cont(c.ctx, k_perm), q_perm);
            float scale = 1.0f / sqrtf((float)d_k);
            kq = ggml_scale(c.ctx, kq, scale);
            kq = ggml_diag_mask_inf(c.ctx, kq, 0);
            struct ggml_tensor * attn = ggml_soft_max(c.ctx, kq);

            struct ggml_tensor * attn_out = ggml_mul_mat(c.ctx, ggml_cont(c.ctx, v_perm), attn);
            attn_out = ggml_permute(c.ctx, attn_out, 0, 2, 1, 3);
            attn_out = ggml_reshape_3d(c.ctx, ggml_cont(c.ctx, attn_out), embed_dim, seq_len, 1);

            struct ggml_tensor * attn_proj = ggml_add(c.ctx, ggml_mul_mat(c.ctx, h->trans_wo[i], attn_out), h->trans_bo[i]);
            x = ggml_add(c.ctx, x, attn_proj);

            struct ggml_tensor * norm2 = ggml_rms_norm(c.ctx, x, 1e-5f);
            norm2 = ggml_add(c.ctx, ggml_mul(c.ctx, norm2, h->trans_ffn_norm_w[i]), h->trans_ffn_norm_b[i]);

            struct ggml_tensor * ffn_up = ggml_mul_mat(c.ctx, h->trans_w1[i], norm2);
            ffn_up = ggml_add(c.ctx, ffn_up, h->trans_b1[i]);
            ffn_up = ggml_silu(c.ctx, ffn_up);

            struct ggml_tensor * ffn_out = ggml_add(c.ctx, ggml_mul_mat(c.ctx, h->trans_w2[i], ffn_up), h->trans_b2[i]);
            x = ggml_add(c.ctx, x, ffn_out);
        }

        x = ggml_rms_norm(c.ctx, x, 1e-5f);
        x = ggml_add(c.ctx, ggml_mul(c.ctx, x, h->trans_final_norm_w), h->trans_final_norm_b);

        struct ggml_tensor * logits = ggml_add(c.ctx, ggml_mul_mat(c.ctx, h->trans_lm_head, x), h->trans_lm_head_b);

        logits = ggml_view_3d(c.ctx, logits, codebook_size, 1, 1, logits->nb[1], logits->nb[2], (seq_len - 1) * logits->nb[1]);

        struct ggml_tensor * logits_1d = ggml_reshape_1d(c.ctx, logits, codebook_size);
        ggml_set_output(logits_1d);

        ggml_build_forward_expand(gf, logits_1d);

        if (!compute_alloc(&c, h->vk_backend)) {
            free(tokens_cpu); free(pos_cpu); compute_end(&c);
            free(logits_buf); free(codes);
            return;
        }

        ggml_backend_tensor_set(tokens, tokens_cpu, 0, seq_len * sizeof(int));
        ggml_backend_tensor_set(pos_ids, pos_cpu, 0, seq_len * sizeof(int));

        ggml_backend_graph_compute(h->vk_backend, gf);

        ggml_backend_tensor_get(logits_1d, logits_buf, 0, codebook_size * sizeof(float));

        int next_token = sample_token_temperature(logits_buf, codebook_size, temperature);
        codes[current_len] = next_token;
        current_len++;

        free(tokens_cpu);
        free(pos_cpu);
        compute_end(&c);
    }

    // Step 3: Decode codes through VQ-VAE decoder
    int out_z_len = current_len;
    float * quantized_z = (float *)calloc(out_z_len * latent_dim, sizeof(float));

    int cb_nbytes = (int)ggml_nbytes(h->codebook);
    float * cb_cpu = (float *)malloc(cb_nbytes);
    ggml_backend_tensor_get(h->codebook, cb_cpu, 0, cb_nbytes);

    for (int i = 0; i < out_z_len; i++) {
        int c = codes[i];
        for (int k = 0; k < latent_dim; k++) {
            quantized_z[i * latent_dim + k] = cb_cpu[c * latent_dim + k];
        }
    }

    eve_decoder_forward(h, quantized_z, 1, out_z_len, output_audio, output_capacity);

    free(quantized_z);
    free(cb_cpu);
    free(logits_buf);
    free(codes);
}

// ============================================================
// Decoder forward (GPU)
// ============================================================

void eve_decoder_forward(eve_handle * h, const float * z, int batch, int seq_len,
                         float * audio, int audio_capacity)
{
    size_t meta_mem = TENSOR_OVERHEAD() * 1000 + GRAPH_OVERHEAD(GGML_DEFAULT_GRAPH_SIZE);
    struct compute_ctx c = compute_begin(h->vk_backend, meta_mem, GGML_DEFAULT_GRAPH_SIZE);

    struct ggml_tensor * inp = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F32, seq_len, h->latent_dim, batch);
    struct ggml_tensor * cur = ggml_conv1d_bias(c.ctx, inp, h->dec_proj, h->dec_proj_bias, PROJ_STRIDE, 1);
    int cur_len = seq_len;

    for (int i = 0; i < NUM_DOWNSAMPLING; i++) {
        int out_ch = (1 << (NUM_DOWNSAMPLING - i - 1));
        cur = ggml_upsample_conv_bias(c.ctx, cur, h->dec_tconv[i], h->dec_tconv_bias[i]);
        cur_len = cur_len * 2;
        cur = ggml_residual_block(c.ctx, cur, out_ch,
                                  h->dec_res_conv1[i], h->dec_res_bias1[i],
                                  h->dec_res_conv2[i], h->dec_res_bias2[i]);
    }

    cur = ggml_conv1d_bias(c.ctx, cur, h->dec_out, h->dec_out_bias, PROJ_STRIDE, 1);

    int out_len = cur_len;
    int needed = batch * 1 * out_len;
    if (needed > audio_capacity) {
        compute_end(&c);
        return;
    }

    ggml_set_output(cur);
    ggml_build_forward_expand(c.gf, cur);

    if (!compute_alloc(&c, h->vk_backend)) {
        compute_end(&c);
        return;
    }

    ggml_backend_tensor_set(inp, z, 0, batch * h->latent_dim * seq_len * sizeof(float));

    if (!compute_exec(&c, h->vk_backend)) {
        compute_end(&c);
        return;
    }

    ggml_backend_tensor_get(cur, audio, 0, needed * sizeof(float));
    compute_end(&c);
}
