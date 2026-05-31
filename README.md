# Eve — Audio-Native Transformer

Train a causal transformer on discrete audio tokens, analogous to how LLMs train on text tokens. No text, no TTS, no STT — raw audio in, raw audio out.

## Architecture

### VQ-VAE (Audio Tokenizer)

Converts raw audio waveforms into discrete codes, then decodes codes back to audio.

```
Audio (float32) → Encoder → Vector Quantizer → Discrete Codes → Decoder → Audio (float32)
```

- **Encoder**: 4× downsampling via Conv1d (stride 2) + ResidualBlock(GELU), then project to latent dim
- **Vector Quantizer**: Nearest-neighbor lookup into a learnable codebook. Standard VQ loss (MSE reconstruction + commitment loss with straight-through gradient estimator)
- **Decoder**: 4× upsampling via ConvTranspose1d (stride 2) + ResidualBlock(GELU), then project back to 1 channel
- **Downsampling factor**: 2^4 = 16×. 24 kHz audio → 1500 tokens/sec
- **Current defaults**: codebook=1024, latent_dim=64, ~1.2M params

### Audio Transformer

Causal (decoder-only) transformer that predicts the next audio token.

```
Codes → Embedding + PositionEncoding → TransformerEncoder(causal) → LayerNorm → Linear(vocab)
```

- Uses TorchSharp's built-in `TransformerEncoderLayer` stacked via `TransformerEncoder`
- Causal mask applied through the encoder's `src_mask` parameter
- Autoregressive generation with temperature sampling
- **Current defaults**: 512-dim, 8 heads, 6 layers, 2048 FFN, ~21M params

### Training

Two stages:

1. **VQ-VAE** — reconstruction loss + commitment loss. Uses **EMA codebook update** (exponential moving average of encoder outputs per code) instead of gradient-based codebook training. Configurable decay (0.99) and commitment cost (0.25). Adam 1e-4, gradient clipping.
2. **Transformer** — cross-entropy on next-code prediction. Adam 1e-4, gradient clipping. VQ-VAE frozen (eval mode). Codes pre-encoded in a `no_grad()` block.

Both stages support per-epoch validation (configurable `ValSplit` ratio) and periodic checkpointing.

## Project Structure

```
Eve/
├── Eve.csproj              # .NET 10 project, TorchSharp 0.103.0
├── Program.cs               # Entry point, training orchestration
├── Utils/
│   ├── Config.cs            # All hyperparameters in one place
│   └── AudioIO.cs           # raw .wav read/write (no libsndfile dependency)
├── Data/
│   └── AudioDataset.cs      # Loads .wav, resamples, pads/crops to fixed length
├── Models/
│   ├── VQVAE.cs             # Encoder + VectorQuantizer + Decoder
│   └── AudioTransformer.cs  # Causal transformer over audio token sequences
├── Training/
│   └── Trainer.cs           # Two-stage training loop
└── README.md
```

## Running

1. Put 24 kHz mono .wav files in `data/`
2. `dotnet run`
3. After training, `generated.wav` is produced (prompts from first file, generates 100 tokens)

Resume from checkpoint: `dotnet run -- --resume 10` (loads epoch 10 checkpoints from `checkpoints/`).

**Current platform**: Linux x64, CPU-only (libtorch-cpu-linux-x64). Builds and runs on .NET 10.

## Hardware Notes

| Hardware | Use |
|----------|-----|
| Strix Halo 128 GB | Development, data preprocessing, CPU training of small models |
| 4070 TI Super 16 GB | Real GPU training — add `TorchSharp-cuda-linux` package |

Switch to CUDA:
```xml
<!-- Remove this -->
<PackageReference Include="libtorch-cpu-linux-x64" Version="2.10.0" />
<!-- Add this -->
<PackageReference Include="TorchSharp-cuda-linux" Version="0.107.0" />
```

## What's Next

### Known Limitations / Immediate Improvements

- **VQ-VAE uses single-level quantization** — no Residual Vector Quantization (RVQ). The `NumQuantizers` config exists but isn't wired. Multiple quantizer levels dramatically improve reconstruction quality at low bitrates.
- **Transformer predicts raw code indices** — not flattened RVQ codes. With RVQ, you'd either flatten all levels across the sequence dimension or use a separate prediction head per level.
- **Learning rate schedule in Config but not wired** — `WarmupSteps` field exists, the scheduler isn't implemented in `Trainer.cs`. Add cosine decay with linear warmup.
- **No proper data split** — `ValSplit` is defined but the current `AudioDataset` re-reads the same files for both train and val. Need a proper split by file index.
- **Dataset assumes all files are the same sample rate** — resampling implemented but untested.
- **Dataset loads all files by index on the fly** — no caching or shuffling. For real data sizes, implement a shuffled `DataLoader`.
- **AudioIO.WriteWav uses `short` data type** — 16-bit output. For high quality, consider 32-bit float WAV.
- **Transformer is a vanilla encoder stack used causally** — using `TransformerEncoder` with a mask works but isn't as efficient as a proper decoder-only architecture. Consider using `TransformerDecoderLayer` directly or writing a custom attention layer with fused KV-cache.

### Path to a Real Model

1. **Dataset**: Collect hours of diverse audio. Eve's current data pipeline works but is naive. Add shuffle, multiprocess loading, and on-the-fly augmentation (noise, pitch shift, tempo).

2. **VQ-VAE quality**: Increase latent dim, add RVQ, increase downsampling ratio. EMA codebook update is already wired. Reference: EnCodec / DAC / MOSS CAT architectures.

3. **Model scale**: Increase transformer depth/width. 22M params is tiny. Target 300M-1B for interesting generation. Scale gradually, monitoring loss curves.

4. **Training stability**: LR warmup + cosine schedule (config fields exist, need trainer wiring), weight decay, AdamW instead of Adam. Gradient clipping already wired.

5. **Evaluation**: Implement objective metrics (SI-SNR, PESQ for speech, FAD for audio quality) and log to TensorBoard.

6. **Sampling improvements**: Top-k / top-p filtering, repetition penalty, KV-cache for faster autoregressive generation.

7. **Multi-GPU**: DeepSpeed-style sharding across multiple 4070s or A100s if you get access. The current TorchSharp path supports `DistributedDataParallel` via `torch.distributed`.

### Switching to a Different Tokenizer

The VQ-VAE is your own implementation. If you find DAC or MOSS CAT works better, port it to C# via TorchSharp. The interface to preserve:

```csharp
// What the transformer needs:
Tensor Encode(Tensor audio) → returns codes shaped [batch, seq_len]
Tensor DecodeFromCodes(Tensor codes) → returns audio shaped [batch, 1, samples]
```

### LLM-Style Scaling

Eve's transformer is the same architecture as GPT-2, just on audio tokens. Everything that works for text LLMs applies:

- Scale compute (model size × data size)
- Scale data (more hours, higher quality, more diverse)
- Scale training (more steps, larger batch, better optimizers)

The 4070 TI Super (16 GB) can probably train 50-100M params with batch size 1-2. The Strix Halo (128 GB unified) can hold larger models but trains slower.
