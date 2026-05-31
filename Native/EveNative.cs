using System.Runtime.InteropServices;

namespace Eve.Native;

public sealed class EveNativeHandle : IDisposable
{
    private IntPtr _handle;
    private bool _disposed;

    public EveNativeHandle(int codebookSize, int latentDim, int embedDim, int numHeads, int numLayers, int feedForwardDim, int maxSeqLen, int sampleRate, int audioLen)
    {
        _handle = eve_create(codebookSize, latentDim, embedDim, numHeads, numLayers, feedForwardDim, maxSeqLen, sampleRate, audioLen);
        if (_handle == IntPtr.Zero)
            throw new InvalidOperationException("eve_create returned null");
    }

    public void InitWeights(uint seed) => eve_init_weights(_handle, seed);
    public void SetLearningRate(float lr) => eve_set_learning_rate(_handle, lr);
    public void SetCommitmentCost(float cost) => eve_set_commitment_cost(_handle, cost);
    public void SetEmaDecay(float decay) => eve_set_ema_decay(_handle, decay);
    public void SaveWeights(string path) { CheckResult(eve_save_weights(_handle, path)); }
    public void LoadWeights(string path) { CheckResult(eve_load_weights(_handle, path)); }

    public int EncoderForward(float[] audio, int batch, int audioLen, float[] z, int zCapacity)
    {
        int result = eve_encoder_forward(_handle, audio, batch, audioLen, z, zCapacity);
        if (result < 0) throw new InvalidOperationException("eve_encoder_forward failed");
        return result;
    }

    public int Encode(float[] audio, int batch, int audioLen, int[] codes, int codesCapacity)
    {
        int result = eve_encode(_handle, audio, batch, audioLen, codes, codesCapacity);
        if (result < 0) throw new InvalidOperationException("eve_encode failed");
        return result;
    }

    public void Quantize(float[] z, int batch, int seqLen, int[] codes, float[] quantized, int zCapacity)
    {
        eve_quantize(_handle, z, batch, seqLen, codes, quantized, zCapacity);
    }

    public void DecoderForward(float[] z, int batch, int seqLen, float[] audio, int audioCapacity)
    {
        eve_decoder_forward(_handle, z, batch, seqLen, audio, audioCapacity);
    }

    public float TrainVqVaeStep(float[] audio, int batch, int audioLen)
    {
        return eve_train_vqvae_step(_handle, audio, batch, audioLen);
    }

    public float TrainTransformerStep(int[] codes, int batch, int seqLen)
    {
        return eve_train_transformer_step(_handle, codes, batch, seqLen);
    }

    public int Generate(float[] promptAudio, int promptLen, int numTokens, float temperature, float[] outputAudio, int outputCapacity)
    {
        // Encode prompt to get latent codes
        int promptZLen = promptLen / 16;
        float[] z = new float[promptZLen * 64];
        int encZLen = EncoderForward(promptAudio, 1, promptLen, z, z.Length);
        
        // Quantize to get code indices
        int[] codes = new int[encZLen];
        float[] qz = new float[encZLen * 64];
        Quantize(z, 1, encZLen, codes, qz, qz.Length);
        
        // Total output codes = prompt codes + generated tokens
        int totalCodes = encZLen + numTokens;
        int outputLen = totalCodes * 16;
        
        // Ensure output buffer is large enough
        if (outputCapacity < outputLen)
        {
            throw new ArgumentException($"Output buffer too small. Need {outputLen}, have {outputCapacity}");
        }
        
        eve_generate(_handle, promptAudio, promptLen, numTokens, temperature, outputAudio, outputCapacity);
        return outputLen;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            if (_handle != IntPtr.Zero)
            {
                eve_destroy(_handle);
                _handle = IntPtr.Zero;
            }
            _disposed = true;
        }
    }

    private static void CheckResult(int ret)
    {
        if (ret != 0) throw new InvalidOperationException($"Native call returned {ret}");
    }

    [DllImport("libeve_native.so", CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr eve_create(int codebook_size, int latent_dim, int embed_dim, int num_heads, int num_layers, int feed_forward_dim, int max_seq_len, int sample_rate, int audio_len);

    [DllImport("libeve_native.so", CallingConvention = CallingConvention.Cdecl)]
    private static extern void eve_destroy(IntPtr h);

    [DllImport("libeve_native.so", CallingConvention = CallingConvention.Cdecl)]
    private static extern void eve_init_weights(IntPtr h, uint seed);

    [DllImport("libeve_native.so", CallingConvention = CallingConvention.Cdecl)]
    private static extern int eve_save_weights(IntPtr h, string path);

    [DllImport("libeve_native.so", CallingConvention = CallingConvention.Cdecl)]
    private static extern int eve_load_weights(IntPtr h, string path);

    [DllImport("libeve_native.so", CallingConvention = CallingConvention.Cdecl)]
    private static extern int eve_encoder_forward(IntPtr h, float[] audio, int batch, int audio_len, float[] z, int z_capacity);

    [DllImport("libeve_native.so", CallingConvention = CallingConvention.Cdecl)]
    private static extern void eve_quantize(IntPtr h, float[] z, int batch, int seq_len, int[] codes, float[] quantized, int z_capacity);

    [DllImport("libeve_native.so", CallingConvention = CallingConvention.Cdecl)]
    private static extern void eve_decoder_forward(IntPtr h, float[] z, int batch, int seq_len, float[] audio, int audio_capacity);

    [DllImport("libeve_native.so", CallingConvention = CallingConvention.Cdecl)]
    private static extern int eve_encode(IntPtr h, float[] audio, int batch, int audio_len, int[] codes, int codes_capacity);

    [DllImport("libeve_native.so", CallingConvention = CallingConvention.Cdecl)]
    private static extern float eve_train_vqvae_step(IntPtr h, float[] audio, int batch, int audio_len);

    [DllImport("libeve_native.so", CallingConvention = CallingConvention.Cdecl)]
    private static extern float eve_train_transformer_step(IntPtr h, int[] codes, int batch, int seq_len);

    [DllImport("libeve_native.so", CallingConvention = CallingConvention.Cdecl)]
    private static extern void eve_set_learning_rate(IntPtr h, float lr);

    [DllImport("libeve_native.so", CallingConvention = CallingConvention.Cdecl)]
    private static extern void eve_set_commitment_cost(IntPtr h, float cost);

    [DllImport("libeve_native.so", CallingConvention = CallingConvention.Cdecl)]
    private static extern void eve_set_ema_decay(IntPtr h, float decay);

    [DllImport("libeve_native.so", CallingConvention = CallingConvention.Cdecl)]
    private static extern void eve_generate(IntPtr h, float[] prompt_audio, int prompt_len, int num_tokens, float temperature, float[] output_audio, int output_capacity);
}
