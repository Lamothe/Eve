using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Eve.Models;

public class AudioTransformer : nn.Module<Tensor, Tensor>
{
    private readonly Embedding _tokenEmbedding;
    private readonly Embedding _positionEmbedding;
    private readonly TransformerEncoder _encoder;
    private readonly LayerNorm _finalNorm;
    private readonly Linear _outputProjection;
    private readonly long _embedDim;

    public AudioTransformer(
        long vocabSize = 1024,
        long embedDim = 512,
        long numHeads = 8,
        long numLayers = 6,
        long feedForwardDim = 2048,
        long maxSeqLen = 2048) : base("AudioTransformer")
    {
        _embedDim = embedDim;

        _tokenEmbedding = Embedding(vocabSize, embedDim);
        _positionEmbedding = Embedding(maxSeqLen, embedDim);

        var encoderLayer = TransformerEncoderLayer(embedDim, numHeads, feedForwardDim, 0.1, Activations.GELU);
        _encoder = TransformerEncoder(encoderLayer, numLayers);

        _finalNorm = LayerNorm(embedDim, 1e-5, true, true, device: null);
        _outputProjection = Linear(embedDim, vocabSize, true);

        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        var batchSize = x.shape[0];
        var seqLen = x.shape[1];

        var tokenEmb = _tokenEmbedding.forward(x.reshape(-1));
        tokenEmb = tokenEmb.reshape(batchSize, seqLen, _embedDim);

        var positions = arange(seqLen, device: x.device).unsqueeze(0).expand(batchSize, seqLen);
        var posEmb = _positionEmbedding.forward(positions);

        var h = tokenEmb + posEmb;
        var mask = CausalMask(seqLen, x.device);
        h = _encoder.call(h, mask, null);
        h = _finalNorm.forward(h);

        return _outputProjection.forward(h);
    }

    private static Tensor CausalMask(long n, Device device)
    {
        var m = full([n, n], double.NegativeInfinity, device: device);
        for (int i = 0; i < n; i++)
            for (int j = 0; j <= i; j++)
                m[i, j] = 0;
        return m;
    }

    public Tensor Generate(Tensor prompt, long steps, double temperature = 1.0)
    {
        var cur = prompt;
        for (int i = 0; i < steps; i++)
        {
            var logits = forward(cur);
            var next = multinomial(functional.softmax(logits[.., -1, ..] / temperature, -1), 1);
            cur = cat([cur, next], 1);
        }
        var keep = (int)prompt.shape[1];
        return cur[.., keep..];
    }
}
