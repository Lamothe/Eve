# Eve — Audio-Native Transformer

Train a causal transformer on discrete audio tokens, analogous to how LLMs train on text tokens. No text, no TTS, no STT — raw audio in, raw audio out.

## Architecture

### VQ-VAE (Audio Tokenizer)

Converts raw audio waveforms into discrete codes, then decodes codes back to audio.

```
Audio (float32) → Encoder → Vector Quantizer → Discrete Codes → Decoder → Audio (float32)
```

- **Encoder**: 4× downsampling via Conv1d (stride 2) + ResidualBlock(GELU), then project to latent dim
- **Vector Quantizer**: Nearest-neighbor lookup into a learnable codebook. EMA codebook update with configurable decay (0.99), values clamped to ±5
- **Decoder**: 4× upsampling via ConvTranspose1d (stride 2) + ResidualBlock(GELU), then project back to 1 channel
- **Downsampling factor**: 2^4 = 16×. 24 kHz audio → 1500 tokens/sec
- **Current defaults**: codebook=256, latent_dim=64

### Transformer

Causal (decoder-only) transformer that predicts the next audio token.

```
Codes → Embedding + PositionEncoding → TransformerEncoder(causal) → LayerNorm → Linear(vocab)
```

- Built with **ggml/llama.cpp** native backend — runs on GPU via Vulkan
- 512-dim embedding, 8 heads, 6 layers, 2048 FFN dim
- Causal mask applied per-layer
- AdamW optimizer with full backward pass on GPU

## Project Structure

```
Eve/
├── Eve.csproj                  # .NET 10 project, C# P/Invoke wrapper
├── Program.cs                  # Entry point, CLI flags, 3-phase training pipeline
├── Utils/
│   ├── Config.cs               # Hyperparameters
│   └── AudioIO.cs              # WAV read/write (pure C#, no native bindings)
├── Data/
│   ├── AudioDataset.cs         # Loads .wav files, zero-pads, serves batches
│   └── SyntheticDataGenerator.cs # KokoroSharp-based synthetic audio generation
├── Native/
│   └── EveNative.cs            # C# P/Invoke wrapper for libeve_native.so
├── Eve.Native/                 # Native library (C++ / ggml / Vulkan)
│   ├── CMakeLists.txt
│   └── src/
│       └── eve_native.cpp      # Encoder, Decoder, Transformer, AdamW, Vulkan backend
├── lib/                        # Built native libraries (gitignored)
├── output/                     # Trained weights (gitignored)
├── data/                       # Training audio (.wav, gitignored)
└── README.md
```

## Running

### Prerequisites

- Linux with AMD GPU supporting Vulkan (RADV driver)
- .NET 10 SDK
- 24 kHz mono .wav files in `data/` (or generate them)

### Generate Training Data

```bash
dotnet run -- --generate 100   # Generate 100 synthetic .wav files
```

### Train (VQ-VAE + Transformer)

```bash
dotnet run                     # Full 3-phase pipeline
```

The training pipeline runs three phases:

1. **Phase 0**: Encode all audio files → `data/codes.bin` (discrete token sequences)
2. **Phase 1**: Train VQ-VAE (5000 steps, AdamW, GPU) — saves to `output/eve_vqvae_trained.bin`
3. **Phase 2**: Train Transformer (10000 steps, AdamW, GPU) — saves to `output/eve_transformer_trained.bin`

### Resume Training

```bash
dotnet run -- --resume         # Load existing weights, continue training
```

If existing weights are found in `output/`, training continues from there instead of starting from scratch.

### Generate Audio

```bash
dotnet run -- --generate-audio 1500 0.9     # Generate from scratch, 1500 tokens, temp=0.9
dotnet run -- --generate-audio 1500 0.9 prompt.wav  # Condition on prompt audio
dotnet run -- --generate-audio 1500 0.9 prompt.wav output.wav  # Save to specific path
```

## CLI Flags

| Flag | Description |
|------|-------------|
| `--generate <N>` | Generate N synthetic .wav files (female voices, random prompts) |
| `--generate-audio <tokens> <temp> [prompt.wav] [output.wav]` | Generate audio from trained model |
| `--resume` | Continue training from existing weights in `output/` |
| `--test-native` | Run P/Invoke wrapper tests |

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | C# / .NET 10 |
| Native Backend | ggml/llama.cpp with Vulkan |
| GPU Acceleration | AMD RADV (Strix Halo: RADV STRIX_HALO) |
| Optimizer | AdamW (F32 precision) |
| Audio I/O | Pure C# WAV reader/writer |

## Current Status

| Phase | Status | Time |
|-------|--------|------|
| Phase 0: Encoding | ✅ Working | ~16s for 246 files |
| Phase 1: VQ-VAE Training | ✅ Working | ~47s (5000 steps) |
| Phase 2: Transformer Training | ✅ Working | ~15min (10000 steps) |
| Audio Generation | ✅ Working | ~1 token/sec |

**Training results** (on 246 synthetic audio files):
- VQ-VAE loss converged to ~0.02 after 5000 steps
- Transformer loss dropped from ~8.3 (random) to ~5.55 after 10000 steps
- Weights saved to `output/` directory (gitignored)

## Hardware

| Hardware | Platform |
|----------|----------|
| Strix Halo (AMD) | Primary dev platform — 128 GB RAM, RADV STRIX_HALO Vulkan |
| NVIDIA CUDA | Not yet tested — would require CUDA backend in ggml |

## How It Works

### Tokenization (VQ-VAE)

```
Audio (24000 samples) → Encoder (4× Conv1d + ResBlocks) → [64-dim latent] → Quantize → Code index
```

Each 1-second audio clip produces ~1500 discrete code indices. The codebook maps each index back to a 64-dimensional vector, which the decoder reconstructs into audio.

### Next-Token Prediction (Transformer)

```
[code0, code1, code2, ...] → Embedding + PosEnc → Transformer blocks → Logits → Sample next code
```

The Transformer autoregressively predicts the next code token given all previous codes. During generation, each step builds a forward graph for the full sequence (without KV-cache), samples the last position's logits with temperature, and appends the token.

### Generation Pipeline

```
Prompt audio → VQ-VAE encode → initial codes → Transformer generate (N tokens) → Decode (VQ-VAE) → Output WAV
```

1. Encode prompt audio through VQ-VAE encoder to get initial codes
2. Autoregressively generate N codes using the Transformer (temperature sampling)
3. Decode the full code sequence through the VQ-VAE decoder
4. Write reconstructed audio to WAV file

## What's Next

### Immediate Improvements

- **KV-Cache**: Add KV-cache support to Transformer for O(1) generation per token instead of O(n)
- **Top-k / Top-p sampling**: Add sampling strategies beyond temperature for more controlled generation
- **Repetition penalty**: Prevent repetitive patterns in long generations
- **Batch generation**: Generate multiple sequences in parallel
- **Training visualization**: TensorBoard or Weights & Biases logging

### Path to Production Quality

1. **Dataset**: Collect hours of diverse, high-quality audio data
2. **VQ-VAE quality**: Increase latent dim, add Residual Vector Quantization (RVQ)
3. **Model scale**: Increase transformer depth/width (300M-1B params target)
4. **Training stability**: Implement LR warmup + cosine decay schedule
5. **Evaluation**: SI-SNR, PESQ, FAD metrics for audio quality
6. **Multi-GPU**: Distributed training across multiple GPUs

## Troubleshooting

### `ggml_new_object: not enough space in the context's memory pool`

The Transformer context needs more tensor slots. Edit `TENSOR_OVERHEAD() * 10000` in `eve_native.cpp` and increase the multiplier.

### Vulkan backend not found

Ensure `libggml-vulkan.so` and its dependencies are in the same directory as `libeve_native.so`. The `.csproj` copies them automatically during build.

### No audio files in `data/`

Generate synthetic data first: `dotnet run -- --generate 100`

### Generated audio sounds noisy

- Increase temperature toward 0.5-0.7 for more conservative generation
- Train for more steps (increase `transSteps` in `Program.cs`)
- Add a prompt audio clip to guide the generation
