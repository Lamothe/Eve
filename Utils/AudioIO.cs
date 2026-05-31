using TorchSharp;
using static TorchSharp.torch;

namespace Eve.Utils;

public static class AudioIO
{
    public static (Tensor waveform, int sampleRate) ReadWav(string path)
    {
        var bytes = File.ReadAllBytes(path);
        using var stream = new MemoryStream(bytes);
        using var reader = new BinaryReader(stream);

        var riff = new string(reader.ReadChars(4));
        if (riff != "RIFF") throw new Exception("Not a WAV file");

        reader.ReadInt32();
        var wave = new string(reader.ReadChars(4));
        if (wave != "WAVE") throw new Exception("Not a WAV file");

        var fmt = new string(reader.ReadChars(4));
        if (fmt != "fmt ") throw new Exception("No fmt chunk");

        var fmtLen = reader.ReadInt32();
        var audioFormat = reader.ReadInt16();
        var numChannels = reader.ReadInt16();
        var sampleRate = reader.ReadInt32();
        reader.ReadInt32();
        reader.ReadInt16();
        var bitsPerSample = reader.ReadInt16();

        if (fmtLen > 16) reader.ReadBytes((int)fmtLen - 16);

        while (new string(reader.ReadChars(4)) != "data") { }

        var dataSize = reader.ReadInt32();
        var rawData = reader.ReadBytes(dataSize);

        Tensor samples;
        if (bitsPerSample == 16)
        {
            var shorts = new short[dataSize / 2];
            Buffer.BlockCopy(rawData, 0, shorts, 0, rawData.Length);
            samples = tensor(shorts, float32).div(32768f);
        }
        else if (bitsPerSample == 32)
        {
            var floats = new float[dataSize / 4];
            Buffer.BlockCopy(rawData, 0, floats, 0, rawData.Length);
            samples = tensor(floats, float32);
        }
        else
        {
            throw new Exception($"Unsupported bits per sample: {bitsPerSample}");
        }

        if (numChannels > 1)
        {
            samples = samples.reshape(new long[] { (long)numChannels, -1L }).mean(new long[] { 0 });
        }

        return (samples.reshape(1, -1), sampleRate);
    }

    public static void WriteWav(string path, float[] samples, int sampleRate)
    {
        var max = 0f;
        for (int i = 0; i < samples.Length; i++)
        {
            var abs = Math.Abs(samples[i]);
            if (abs > max) max = abs;
        }
        if (max < 1e-8f) max = 1f;
        var data = new byte[samples.Length * 2];
        for (int i = 0; i < samples.Length; i++)
        {
            var s = (short)(samples[i] / max * 32767f);
            data[i * 2] = (byte)(s & 0xFF);
            data[i * 2 + 1] = (byte)((s >> 8) & 0xFF);
        }
        WriteRawWav(path, data, sampleRate);
    }

    public static void WriteWav(string path, Tensor waveform, int sampleRate)
    {
        var mono = waveform.squeeze();
        if (mono.dim() > 1) mono = mono.mean(new long[] { 0 }).reshape(new long[] { -1 });

        var normalized = mono.div(mono.abs().max() + 1e-8f).clamp(-1f, 1f);
        var shorts = normalized.mul(32767f).to(int16);

        var shortsArr = shorts.data<short>().ToArray();
        var data = new byte[shortsArr.Length * 2];
        Buffer.BlockCopy(shortsArr, 0, data, 0, data.Length);

        WriteRawWav(path, data, sampleRate);
    }

    private static void WriteRawWav(string path, byte[] data, int sampleRate)
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream);

        var dataSize = data.Length;
        var fileSize = 36 + dataSize;

        writer.Write("RIFF".ToCharArray());
        writer.Write(fileSize);
        writer.Write("WAVE".ToCharArray());
        writer.Write("fmt ".ToCharArray());
        writer.Write(16);
        writer.Write((short)1);
        writer.Write((short)1);
        writer.Write(sampleRate);
        writer.Write(sampleRate * 2);
        writer.Write((short)2);
        writer.Write((short)16);
        writer.Write("data".ToCharArray());
        writer.Write(dataSize);
        writer.Write(data);

        File.WriteAllBytes(path, stream.ToArray());
    }
}
