# Eve — Voice-Native Conversational Agent

**Status: Building — Vulkan GPU training on AMD Strix Halo**

## Vision

Eve is a voice-native AI agent that engages in natural, fluid conversation. No text. No turn-taking tokens. No robotic Q&A loops. Just audio in, audio out.

**Core principles:**
- **Voice-native**: All processing is audio. Text is poison to natural conversation — it strips tone, timing, overlap, and emotion.
- **Always listening**: Continuous audio processing, not discrete chunks. Like a human, Eve stays "in the room."
- **Implicit speaker recognition**: Speaker identity is learned from voice characteristics alone — no labels, no IDs. Eve determines who's talking from how they sound.
- **Voice flexibility**: Eve can adopt any voice style (Southern drawl, New York cabbie, young British woman) from a short audio sample. Voice is a conditioning vector, not a fixed trait.

## Architecture (Planned)

### Dual-Stream Conditioning Model

```
Speaker audio → [Voice Encoder] → style embedding ──┐
                                                     ├──→ [Transformer] → [Audio Decoder] → Eve's response
Conversation audio ──────────────────────────────────┘
```

- **Voice Encoder**: Extracts a style embedding from a short sample of the target voice. Learns to disentangle "what" from "who."
- **Transformer**: Processes the conversation audio, conditioned on the target voice style. Learns turn-taking, interruption, and response timing from data patterns alone.
- **Audio Decoder**: Generates Eve's response in the conditioned voice style.

### Key Design Decisions

1. **No speaker labels**: Voice identity is learned from the audio signal itself
2. **No turn-taking logic**: The model learns when to listen and when to speak from conversational training data
3. **Voice as conditioning**: Any voice can be selected by providing a sample — enables voice flexibility
4. **Continuous audio**: Streaming input/output, not discrete utterances

### Training Data

Conversations with natural dynamics — overlapping speech, interruptions, ambient sounds, silences. Sources: movies, TV shows, podcasts, YouTube conversations. Two different voices in conversation is all that's needed — no labels required.

## Current Implementation (Proof of Concept)

The current codebase implements a causal audio transformer used for infrastructure validation:

```
Audio → STFT → Top-K bins/frame → Token sequence → Transformer → Next-token prediction
```

- **STFT**: 1024 window, 256 hop, 1024 FFT → 513 bins
- **Tokens**: Top-32 bins per frame, sorted by energy, flattened to ~3000 tokens/sec
- **Optimizer**: AdamW with GPU-resident step (`ggml_opt_step_adamw`)
- **Attention**: Multi-head self-attention with causal masking via `ggml_soft_max_ext`
- **Backend**: ggml/llama.cpp with full Vulkan GPU acceleration

### Model Configuration

- **Parameters**: ~2.8M
- **Embedding dimension**: 256
- **Attention heads**: 4 (head dimension = 64)
- **Transformer layers**: 4
- **Feed-forward dimension**: 1024
- **Sequence length**: 128
- **Training epochs**: 50
- **Learning rate**: 1e-7

This is a toy model used to validate the training pipeline. The full architecture will be substantially larger.

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

# Resume from checkpoint
dotnet run -- train <audio_dir> <output_model.bin> --resume <checkpoint>
```

## Vulkan Backend

Training runs entirely on the AMD Radeon 8060S GPU via Vulkan:

- **Cross-entropy loss**: Decomposed into `softmax → log → mul → sum → scale` (all Vulkan-supported ops)
- **Multi-head attention**: llama.cpp's permute-based pattern with `ggml_soft_max_ext` fused masking
- **AdamW update**: Fully GPU-resident via `ggml_opt_step_adamw` — eliminates PCIe bottleneck
- **Computation graph**: Fresh context per training step to avoid memory leaks
- **Checkpointing**: Saves weights + AdamW moments + iteration count for resumable training

## Hardware

- **GPU**: AMD Radeon 8060S (Strix Halo) — RDNA3.5, gfx1151
- **RAM**: 128 GB unified memory
- **Training**: Vulkan backend via ggml/llama.cpp
- **OS**: Fedora 44 with ROCm 7.1.1 userspace packages
