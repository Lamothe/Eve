# Eve — Audio-Native Transformer

**Status: Working — Vulkan GPU training with multi-head self-attention on AMD Strix Halo**

## Architecture (Current — DFT-based)

Audio is tokenized via STFT top-K frequency bin selection, then modeled with a causal transformer that predicts the next DFT bin index.

```
Audio (float32) → STFT → Top-K bins/frame → Flattened token sequence → Causal Transformer → Logits → Sample next bin
```

- **STFT**: 1024 window, 256 hop, 1024 FFT → 513 bins
- **Tokens**: Top-32 bins per frame, flattened to ~3000 tokens/sec
- **Model**: ggml/llama.cpp native backend with Vulkan GPU acceleration
- **Optimizer**: AdamW with GPU-resident optimizer step (`ggml_opt_step_adamw`)
- **Attention**: Multi-head self-attention with causal masking, following llama.cpp's permute-based pattern

### Current Model Configuration

- **Parameters**: ~2.8M
- **Embedding dimension**: 256
- **Attention heads**: 4 (head dimension = 64)
- **Transformer layers**: 4
- **Feed-forward dimension**: 1024
- **Sequence length**: 128
- **Causal masking**: Fused with softmax via `ggml_soft_max_ext`
- **Training epochs**: 50
- **Learning rate**: 1e-7

## Project Structure

```
Eve/
├── Eve.csproj                  # .NET 10 project
├── Program.cs                  # Entry point, CLI: train, generate
├── Utils/
│   ├── STFT.cs                 # STFT/iSTFT with Hann window, overlap-add
│   ├── AudioIO.cs              # WAV read/write (PCM mono)
│   └── Config.cs
├── Data/
│   └── AudioTokenizer.cs       # Encodes audio → top-K bin indices
├── Eve.Native/                 # Native library (C++ / ggml)
│   ├── CMakeLists.txt
│   ├── include/
│   │   └── eve_native.h        # Public C API
│   └── src/
│       └── eve_native.cpp      # Transformer, AdamW, Vulkan training
├── lib/                        # Built native libraries
├── output/                     # Trained weights
├── data/                       # Training audio (.wav)
└── README.md
```

## Building

```bash
# Build native library (requires llama.cpp with Vulkan support)
cmake -S Eve.Native -B Eve.Native/build
cmake --build Eve.Native/build -j$(nproc)

# Build C# project
dotnet build
```

## CLI

```bash
# Train on audio files (uses Vulkan GPU automatically)
dotnet run -- train <audio_dir> <output_model.bin>

# Generate audio from trained model
dotnet run -- generate <model.bin> <prompt.wav> <output.wav> <num_frames> <temperature>
```

## Vulkan Backend

Training runs entirely on the AMD Radeon 8060S GPU via Vulkan:

- **Cross-entropy loss**: Decomposed into `softmax → log → mul → sum → scale` (all Vulkan-supported ops)
- **Multi-head attention**: Implements llama.cpp's permute-based pattern:
  1. QKV projection → reshape `(head_dim, num_heads, seq_len)`
  2. Permute Q,K to `(head_dim, seq_len, num_heads)` so `ggml_mul_mat` broadcasts over heads
  3. Scores: `K^T @ Q`, then fused `scale + mask + softmax` via `ggml_soft_max_ext`
  4. V transpose + matmul for attention output
  5. Permute back + `ggml_cont_2d` to flatten heads
- **Weights & moments**: Allocated on Vulkan backend
- **Computation graph**: Fresh context per training step to avoid memory leaks
- **AdamW update**: Fully GPU-resident using `ggml_opt_step_adamw` from llama.cpp's optimization infrastructure

### Key Insights

1. **ggml_opt_step_adamw**: llama.cpp provides a built-in ggml operation that runs the entire AdamW update on GPU. This eliminates the PCIe bottleneck from CPU-based optimizer updates.

2. **ggml_soft_max_ext**: Fuses scale multiplication, mask addition, and softmax into a single Vulkan kernel, enabling efficient causal attention.

3. **Permute pattern**: llama.cpp's attention uses `ggml_permute(ctx, q, 0, 2, 1, 3)` to swap dims 1 and 2, placing `seq_len` in dim 1. This is critical — it lets `ggml_mul_mat` broadcast the matmul over the head dimension without manual batching.

## Hardware

- **GPU**: AMD Radeon 8060S (Strix Halo) — RDNA3.5, gfx1151
- **RAM**: 128 GB unified memory
- **Training**: Vulkan backend via ggml/llama.cpp
- **OS**: Fedora 44 with ROCm 7.1.1 userspace packages

## Performance

With GPU-resident AdamW:
- **Small model (530K params)**: ~14 minutes for 3 epochs
- **Large model (2.5M params)**: ~10 hours for 50 epochs (estimated)

Without GPU AdamW (CPU-based optimizer):
- **Small model**: ~35 minutes for 3 epochs
- **Large model**: ~26 hours for 50 epochs (estimated)

The 2.5x speedup comes from eliminating ~70 MB/step of PCIe transfers between GPU and CPU.
