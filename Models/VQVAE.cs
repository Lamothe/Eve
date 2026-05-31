using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Eve.Models;

public class ResidualBlock : nn.Module<Tensor, Tensor>
{
    private readonly Sequential _convs;

    public ResidualBlock(long channels, long kernelSize = 3) : base("ResidualBlock")
    {
        _convs = Sequential(
            ("conv1", Conv1d(channels, channels, kernelSize, 1, kernelSize / 2, 1, PaddingModes.Zeros, 1, true, null)),
            ("conv2", Conv1d(channels, channels, kernelSize, 1, kernelSize / 2, 1, PaddingModes.Zeros, 1, true, null))
        );
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        var residual = x;
        x = functional.gelu(_convs.forward(x));
        return x + residual;
    }
}

public class Encoder : nn.Module<Tensor, Tensor>
{
    private readonly Sequential _layers;

    public Encoder(long inChannels, long latentDim, long numDownsampling = 4) : base("Encoder")
    {
        var channels = inChannels;
        var list = new List<(string, nn.Module<Tensor, Tensor>)>();
        for (int i = 0; i < numDownsampling; i++)
        {
            list.Add(($"conv{i}", Conv1d(channels, channels * 2, 4, 2, 1, 1, PaddingModes.Zeros, 1, true, null)));
            list.Add(($"res{i}", new ResidualBlock(channels * 2)));
            channels *= 2;
        }
        list.Add(($"proj", Conv1d(Math.Min(channels, 512), latentDim, 3, 1, 1, 1, PaddingModes.Zeros, 1, true, null)));
        _layers = Sequential(list.ToArray());
        RegisterComponents();
    }

    public override Tensor forward(Tensor x) => _layers.forward(x);
}

public class Decoder : nn.Module<Tensor, Tensor>
{
    private readonly Sequential _layers;

    public Decoder(long latentDim, long outChannels, long numUpsampling = 4) : base("Decoder")
    {
        var channels = latentDim;
        for (int i = 0; i < numUpsampling; i++)
            channels = Math.Min(channels * 2, 512);

        var list = new List<(string, nn.Module<Tensor, Tensor>)>();
        for (int i = 0; i < numUpsampling; i++)
        {
            list.Add(($"tconv{i}", ConvTranspose1d(channels, channels / 2, 4, 2, 1, 0, 1, PaddingModes.Zeros, 1, true, null)));
            list.Add(($"res{i}", new ResidualBlock(channels / 2)));
            channels /= 2;
        }
        list.Add(($"out", Conv1d(channels, outChannels, 3, 1, 1, 1, PaddingModes.Zeros, 1, true, null)));
        _layers = Sequential(list.ToArray());
        RegisterComponents();
    }

    public override Tensor forward(Tensor x) => _layers.forward(x);
}

public class VectorQuantizer : nn.Module<Tensor, Tensor>
{
    private readonly long _codebookSize;
    private readonly long _latentDim;
    private readonly float _decay;
    private readonly float _epsilon;
    private readonly float _commitmentCost;
    private Tensor _codebook;
    private Tensor _emaCount;
    private Tensor _emaSum;

    public VectorQuantizer(
        long codebookSize,
        long latentDim,
        float decay = 0.99f,
        float epsilon = 1e-5f,
        float commitmentCost = 0.25f) : base("VectorQuantizer")
    {
        _codebookSize = codebookSize;
        _latentDim = latentDim;
        _decay = decay;
        _epsilon = epsilon;
        _commitmentCost = commitmentCost;

        _codebook = empty([codebookSize, latentDim]);
        nn.init.uniform_(_codebook, -1.0 / codebookSize, 1.0 / codebookSize);

        _emaCount = zeros([codebookSize]);
        _emaSum = zeros([codebookSize, latentDim]);

        RegisterComponents();
    }

    public (Tensor quantized, Tensor codes, Tensor loss) Quantize(Tensor x, bool useEma = true)
    {
        x = x.permute(0, 2, 1);
        var flat = x.reshape(-1, _latentDim);

        var dist = flat.pow(2).sum(-1, keepdim: true) +
                   _codebook.pow(2).sum(-1) -
                   2 * flat.mm(_codebook.T);

        var indices = dist.argmin(-1);
        var quantized = _codebook.index_select(0, indices.reshape(-1));
        quantized = quantized.reshape(indices.shape[0], indices.shape[1], _latentDim);
        var codes = indices.reshape(x.shape[0], -1);

        if (useEma)
        {
            var loss = nn.functional.mse_loss(quantized.detach(), flat) * _commitmentCost;
            quantized = flat + (quantized - flat).detach();
            quantized = quantized.reshape(x.shape).permute(0, 2, 1);
            return (quantized, codes, loss);
        }
        else
        {
            var loss = nn.functional.mse_loss(quantized, flat.detach()) +
                       nn.functional.mse_loss(quantized.detach(), flat) * _commitmentCost;
            quantized = flat + (quantized - flat).detach();
            quantized = quantized.reshape(x.shape).permute(0, 2, 1);
            return (quantized, codes, loss);
        }
    }

    public void UpdateEma(Tensor encoderOutput)
    {
        using (no_grad())
        {
            var permuted = encoderOutput.permute(0, 2, 1);
            var flat = permuted.reshape(-1, _latentDim);

            var dist = flat.pow(2).sum(-1, keepdim: true) +
                       _codebook.pow(2).sum(-1) -
                       2 * flat.mm(_codebook.T);
            var indices = dist.argmin(-1);

            var n = flat.shape[0];
            var idx2d = indices.reshape(n, 1).to(int64);
            var oneHot = zeros([n, _codebookSize], device: flat.device, dtype: float32);
            oneHot.scatter_add_(1, idx2d, ones([n, 1], device: flat.device, dtype: float32));
            var counts = oneHot.sum(0);
            var sumPerCode = oneHot.T.mm(flat);

            _emaCount.mul_(_decay);
            _emaCount.add_(counts.mul(1.0f - _decay));

            _emaSum.mul_(_decay);
            _emaSum.add_(sumPerCode.mul(1.0f - _decay));

            var nTotal = _emaCount.sum();
            var normCount = (_emaCount + _epsilon) / (nTotal + _codebookSize * _epsilon) * nTotal;
            var newCodebook = _emaSum / normCount.unsqueeze(-1);
            _codebook.copy_(newCodebook);
        }
    }

    public override Tensor forward(Tensor x)
    {
        var (q, _, _) = Quantize(x);
        return q;
    }

    public Tensor DecodeCodes(Tensor codes)
    {
        var flat = _codebook.index_select(0, codes.reshape(-1));
        flat = flat.reshape(codes.shape[0], codes.shape[1], _latentDim);
        return flat.permute(0, 2, 1);
    }
}

public class VQVAE : nn.Module<Tensor, Tensor>
{
    public Encoder EncoderNet { get; }
    public VectorQuantizer Quantizer { get; }
    public Decoder DecoderNet { get; }

    public VQVAE(
        long inChannels = 1,
        long latentDim = 64,
        long codebookSize = 1024,
        float emaDecay = 0.99f,
        float emaEpsilon = 1e-5f,
        float commitmentCost = 0.25f) : base("VQVAE")
    {
        EncoderNet = new Encoder(inChannels, latentDim);
        Quantizer = new VectorQuantizer(codebookSize, latentDim, emaDecay, emaEpsilon, commitmentCost);
        DecoderNet = new Decoder(latentDim, inChannels);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        var (q, _, _) = Encode(x);
        return DecoderNet.forward(q);
    }

    public (Tensor quantized, Tensor codes, Tensor loss) Encode(Tensor x, bool useEma = true)
    {
        var z = EncoderNet.forward(x);
        return Quantizer.Quantize(z, useEma);
    }

    public Tensor GetEncoderOutput(Tensor x) => EncoderNet.forward(x);

    public Tensor DecodeFromCodes(Tensor codes)
    {
        var z = Quantizer.DecodeCodes(codes);
        return DecoderNet.forward(z);
    }
}
