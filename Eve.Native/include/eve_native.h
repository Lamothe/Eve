#ifndef EVE_NATIVE_H
#define EVE_NATIVE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct eve_handle eve_handle;

// Create/destroy the native handle.
// Initializes GGML context and allocates weight tensors for encoder, quantizer,
// decoder, and transformer. Also allocates AdamW moment tensors.
eve_handle* eve_create(
    int codebook_size,
    int latent_dim,
    int embed_dim,
    int num_heads,
    int num_layers,
    int feed_forward_dim,
    int max_seq_len,
    int sample_rate,
    int audio_len);

// Free all memory associated with the handle.
void eve_destroy(eve_handle* h);

// ---- Weight I/O ----
// Save/load all weight tensors to/from a single binary file.
// Format: [num_tensors:4][name_len:4][name:name_len][data:size]...
int eve_save_weights(eve_handle* h, const char* path);
int eve_load_weights(eve_handle* h, const char* path);

// ---- VQ-VAE Encoder ----
// Run the encoder forward pass.
// Input:  audio  shaped [batch, 1, audio_len]  (float32, interleaved, layout NCH)
// Output: z      shaped [batch, latent_dim, out_len]  (caller allocated, float32)
// Returns out_len (number of time steps in encoder output).
int eve_encoder_forward(eve_handle* h, const float* audio, int batch, int audio_len, float* z, int z_capacity);

// ---- Vector Quantizer ----
// Quantize encoder output.
// Input:  z       shaped [batch, latent_dim, seq_len]
// Output: codes   shaped [batch, seq_len] (int32)
// Output: quantized shaped [batch, latent_dim, seq_len] (float32)
void eve_quantize(eve_handle* h, const float* z, int batch, int seq_len, int* codes, float* quantized, int z_capacity);

// ---- VQ-VAE Decoder ----
// Run the decoder forward pass on quantized codes.
// Input:  z       shaped [batch, latent_dim, seq_len]
// Output: audio   shaped [batch, 1, audio_len] (float32)
void eve_decoder_forward(eve_handle* h, const float* z, int batch, int seq_len, float* audio, int audio_capacity);

// ---- Encode full pipeline (encoder + quantize) ----
// One-shot: audio -> codes
int eve_encode(eve_handle* h, const float* audio, int batch, int audio_len, int* codes, int codes_capacity);

// ---- Training steps ----
// VQ-VAE training step. Returns loss value.
float eve_train_vqvae_step(eve_handle* h, const float* audio, int batch, int audio_len);

// Transformer training step. Returns loss value.
float eve_train_transformer_step(eve_handle* h, const int* codes, int batch, int seq_len);

// Set learning rate for all optimizers.
void eve_set_learning_rate(eve_handle* h, float lr);

// Set VQ-VAE commitment cost (default 1.0).
void eve_set_commitment_cost(eve_handle* h, float cost);

// Set EMA decay for codebook update (default 0.99).
void eve_set_ema_decay(eve_handle* h, float decay);

// ---- Generation ----
// Generate audio by autoregressive sampling.
void eve_generate(eve_handle* h, const float* prompt_audio, int prompt_len, int num_tokens, float temperature, float* output_audio, int output_capacity);

// ---- Random initialization ----
// Initialize all weights with small random values (uniform [-1/size, 1/size]).
void eve_init_weights(eve_handle* h, unsigned int seed);

#ifdef __cplusplus
}
#endif

#endif // EVE_NATIVE_H
