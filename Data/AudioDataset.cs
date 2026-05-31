using TorchSharp;
using static TorchSharp.torch;

namespace Eve.Data;

public class AudioDataset
{
    private readonly string[] _files;
    private readonly int _sampleRate;
    private readonly int _targetLength;

    public AudioDataset(string dataPath, int sampleRate, int targetLength)
    {
        _sampleRate = sampleRate;
        _targetLength = targetLength;
        _files = Directory.GetFiles(dataPath, "*.wav", SearchOption.AllDirectories);
    }

    public long Count => _files.Length;

    public Tensor GetAudio(long index)
    {
        var (wav, sr) = Utils.AudioIO.ReadWav(_files[index]);
        if (sr != _sampleRate)
            wav = Resample(wav, sr, _sampleRate);

        if (wav.numel() > _targetLength)
            wav = wav[.., .._targetLength];
        else if (wav.numel() < _targetLength)
            wav = nn.functional.pad(wav, [0, _targetLength - (int)wav.numel()]);

        return wav;
    }

    private static Tensor Resample(Tensor wav, int origSr, int targetSr)
    {
        var ratio = (double)targetSr / origSr;
        var newLen = (long)(wav.numel() * ratio);
        return nn.functional.interpolate(wav.unsqueeze(0), [newLen], null).squeeze(0);
    }
}
