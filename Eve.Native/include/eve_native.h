#ifndef EVE_NATIVE_H
#define EVE_NATIVE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct eve_handle eve_handle;

// Create a new transformer model
eve_handle * eve_create(int vocab_size, int embed_dim, int num_heads, int num_layers,
                        int feed_forward_dim, int max_seq_len);

// Destroy the model and free resources
void eve_destroy(eve_handle * h);

// Initialize weights with random values
void eve_init_weights(eve_handle * h, unsigned int seed);

// Set learning rate for AdamW optimizer
void eve_set_learning_rate(eve_handle * h, float lr);

// Train one step on a sequence of tokens
// Returns the cross-entropy loss
float eve_train_step(eve_handle * h, int * tokens, int seq_len);

// Generate tokens autoregressively
// Returns the number of tokens generated
int eve_generate(eve_handle * h, int * prompt_tokens, int prompt_len,
                 int * output_tokens, int max_output_len, float temperature);

// Save model weights to file (returns 0 on success)
int eve_save_model(eve_handle * h, const char * path);

// Load model weights from file (returns 0 on success)
int eve_load_model(eve_handle * h, const char * path);

#ifdef __cplusplus
}
#endif

#endif // EVE_NATIVE_H
