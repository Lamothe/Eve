# AGENTS.md — Eve Project Conventions

## Project Overview

Eve is a voice-native conversational AI agent. All processing is audio — no text. The goal is natural, fluid conversation with implicit speaker recognition and voice-flexible generation.

The project uses ggml/llama.cpp for the neural network and trains on Vulkan GPU (AMD Radeon 8060S, Strix Halo).

## Build & Test

### Build native library

```bash
cmake -S Eve.Native -B Eve.Native/build
cmake --build Eve.Native/build -j$(nproc)
```

Requires llama.cpp built with Vulkan support at `/home/michael/Projects/llama.cpp`.

### Build C# project

```bash
dotnet build
```

The C# project copies native libraries (libeve_native.so, libggml-*.so) to the output directory on build.

### Train

```bash
dotnet run -- train <audio_dir> <output_model.bin>
dotnet run -- train <audio_dir> <output_model.bin> --resume <checkpoint>
```

### Generate

```bash
dotnet run -- generate <model.bin> <prompt.wav> <output.wav> <num_frames> <temperature>
```

## Architecture

### C# side (`Program.cs`, `Eve.Native/`, `Utils/`, `Data/`)

- Orchestrates training loop, tokenization, and audio I/O
- `AudioTokenizer.cs`: STFT → top-K bin extraction → token sequences
- `STFT.cs`: STFT/iSTFT with Hann window and overlap-add
- `AudioIO.cs`: WAV read/write
- `EveNative.cs`: P/Invoke bindings to the native C library

### C++ side (`Eve.Native/src/eve_native.cpp`)

- `eve_create`: Allocates model weights, moments, causal mask on Vulkan backend
- `eve_train_step`: Forward pass → backward pass → AdamW update (all GPU)
- `transformer_forward`: Multi-head self-attention + FFN blocks
- `eve_generate`: Autoregressive token generation with temperature sampling
- `eve_save_checkpoint` / `eve_load_checkpoint`: Full training state persistence

## Key Patterns

### Vulkan GPU Training

- Weights and moments are allocated on `h->vk_backend`
- Each training step creates a **fresh context** for the computation graph (avoids memory leaks)
- The graph is allocated via `ggml_backend_alloc_ctx_tensors` on Vulkan
- Executed via `ggml_backend_graph_compute(h->vk_backend, gf)`

### Cross-Entropy Loss (Vulkan-Compatible)

`ggml_cross_entropy_loss` is NOT supported on Vulkan. Decompose into:
```
probs = ggml_soft_max(logits)
log_probs = ggml_log(probs)
nll = ggml_mul(log_probs, target_labels)
loss = ggml_sum(nll)
loss = ggml_scale(loss, -1.0f)
```

All of these individual ops are Vulkan-supported.

### Multi-Head Self-Attention (llama.cpp Pattern)

```
1. QKV projection: ggml_mul_mat → shape (embed_dim, seq_len)
2. Reshape: ggml_reshape_3d → (head_dim, num_heads, seq_len)
3. Permute Q,K: ggml_permute(ctx, q, 0, 2, 1, 3) → (head_dim, seq_len, num_heads)
4. Scores: ggml_mul_mat(k, q) → (seq_len, seq_len, num_heads)
5. Causal mask + scale + softmax: ggml_soft_max_ext(scores, mask, scale, 0)
6. V transpose: ggml_cont(ggml_transpose(v)) → (seq_len, head_dim, num_heads)
7. Output: ggml_mul_mat(v_trans, scores) → (head_dim, seq_len, num_heads)
8. Permute back: ggml_permute(0, 2, 1, 3) → (head_dim, num_heads, seq_len)
9. Flatten: ggml_cont_2d → (embed_dim, seq_len)
10. Output projection: ggml_mul_mat(wo, attn_out)
```

The permute `(0, 2, 1, 3)` is critical — it swaps dims 1 and 2 so that `seq_len` is in dim 1, enabling `ggml_mul_mat` to broadcast over heads automatically.

### GPU-Resident AdamW

Uses `ggml_opt_step_adamw(ctx, weight, grad, m, v, params)` — a built-in ggml operation that runs entirely on GPU. Eliminates the ~70 MB/step PCIe bottleneck from CPU-based optimizer updates.

Parameters are a 7-element tensor: `[alpha, beta1, beta2, eps, wd, beta1_correction, beta2_correction]`.

### Checkpointing

`eve_save_checkpoint` saves:
1. Model config (dimensions)
2. AdamW iteration count
3. All weight tensors
4. All moment tensors (m and v)

`eve_load_checkpoint` restores the full training state and returns the iteration count.

### Causal Mask

Pre-allocated during `eve_create` as a `(max_seq_len × max_seq_len)` tensor on Vulkan. Initialized with 0 for allowed positions and `-INFINITY` for future positions. During forward pass, a view into the mask is taken for the current sequence length and made contiguous for `ggml_soft_max_ext`.

## Directory Layout

```
Eve/
├── Eve.csproj                # .NET project
├── Program.cs                # Entry point, training loop
├── Utils/
│   ├── STFT.cs               # STFT/iSTFT
│   ├── AudioIO.cs            # WAV I/O
│   └── Config.cs             # Configuration (unused)
├── Data/
│   └── AudioTokenizer.cs     # Audio → tokens
├── Eve.Native/
│   ├── EveNative.cs          # C# P/Invoke bindings
│   ├── CMakeLists.txt
│   ├── include/
│   │   └── eve_native.h      # C API header
│   └── src/
│       └── eve_native.cpp    # Transformer, training, Vulkan
├── lib/                      # Built .so files
├── data/                     # Training audio (.wav)
└── output/                   # Models, checkpoints, training logs
```

## Design Principles

1. **No text in the pipeline** — text strips natural conversation dynamics
2. **No explicit speaker labels** — voice identity learned from audio characteristics
3. **No hardcoded turn-taking** — model learns timing from data
4. **Voice as conditioning** — any voice style from a short audio sample
5. **Everything on GPU** — forward, backward, and optimizer all run on Vulkan
6. **Fresh context per step** — avoids ggml memory leaks from graph accumulation
