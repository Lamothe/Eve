# Eve — Audio-Native Transformer

**Status: Working — Vulkan GPU training on AMD Strix Halo**

## Architecture (Current — DFT-based)

Audio is tokenized via STFT top-K frequency bin selection, then modeled with a causal transformer that predicts the next DFT bin index.

```
Audio (float32) → STFT → Top-K bins/frame → Flattened token sequence → Causal Transformer → Logits → Sample next bin
```

- **STFT**: 1024 window, 256 hop, 1024 FFT → 513 bins
- **Tokens**: Top-32 bins per frame, flattened to ~3000 tokens/sec
- **Model**: ggml/llama.cpp native backend with Vulkan GPU acceleration
- **Optimizer**: AdamW with GPU-resident optimizer step (ggml_opt_step_adamw)

### Current Model Configuration

- **Parameters**: ~2.5M
- **Embedding dimension**: 256
- **Transformer layers**: 4
- **Feed-forward dimension**: 1024
- **Sequence length**: 128
- **Training epochs**: 50
- **Learning rate**: 0.00001

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
- **Weights & moments**: Allocated on Vulkan backend
- **Computation graph**: Fresh context per training step to avoid memory leaks
- **AdamW update**: Fully GPU-resident using `ggml_opt_step_adamw` from llama.cpp's optimization infrastructure
  - No CPU↔GPU transfers during optimization
  - 2.5x speedup compared to CPU-based AdamW
  - Enables practical training of larger models (~10 hours for 50 epochs on 2.5M params)

### Key Insight

llama.cpp provides `ggml_opt_step_adamw`, a built-in ggml operation that runs the entire AdamW update on GPU. By integrating this into the training graph, we eliminate the PCIe bottleneck that previously made GPU training impractical.

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
