using System;
using System.IO;

namespace Eve.Utils;

public static class AudioIO
{
    public static float[] ReadWav(string path, out int sampleRate)
    {
        using var fs = new FileStream(path, FileMode.Open, FileAccess.Read);
        using var br = new BinaryReader(fs);

        // RIFF Header
        string riff = new string(br.ReadChars(4));
        if (riff != "RIFF") throw new InvalidDataException("Not a valid RIFF file");
        br.ReadInt32(); // Chunk size
        string format = new string(br.ReadChars(4));
        if (format != "WAVE") throw new InvalidDataException("Not a valid WAVE file");

        int channels = 0;
        sampleRate = 0;
        int bitsPerSample = 0;
        byte[]? dataBytes = null;

        while (fs.Position < fs.Length)
        {
            string chunkId = new string(br.ReadChars(4));
            int chunkSize = br.ReadInt32();

            if (chunkId == "fmt ")
            {
                short audioFormat = br.ReadInt16(); // 1 = PCM
                channels = br.ReadInt16();
                sampleRate = br.ReadInt32();
                br.ReadInt32(); // Byte rate
                br.ReadInt16(); // Block align
                bitsPerSample = br.ReadInt16();
                if (chunkSize > 16) br.ReadBytes(chunkSize - 16);
            }
            else if (chunkId == "data")
            {
                dataBytes = br.ReadBytes(chunkSize);
                break; // We found the audio samples, we can stop
            }
            else
            {
                br.ReadBytes(chunkSize); // Skip metadata / LIST / INFO chunks
            }
        }

        if (dataBytes == null) throw new InvalidDataException("No 'data' chunk found in WAV");
        if (bitsPerSample != 16) throw new NotSupportedException("Only 16-bit PCM WAV files are supported");

        int numSamples = dataBytes.Length / 2;
        float[] samples = new float[numSamples];
        for (int i = 0; i < numSamples; i++)
        {
            short s = BitConverter.ToInt16(dataBytes, i * 2);
            samples[i] = s / 32768f; // Normalize to [-1.0, 1.0]
        }

        // Downmix stereo to mono if necessary
        if (channels == 2)
        {
            float[] mono = new float[numSamples / 2];
            for (int i = 0; i < mono.Length; i++)
            {
                mono[i] = (samples[i * 2] + samples[i * 2 + 1]) / 2f;
            }
            return mono;
        }

        return samples;
    }

    public static void WriteWav(string path, float[] samples, int sampleRate)
    {
        short[] pcm = new short[samples.Length];
        for (int i = 0; i < samples.Length; i++)
        {
            float s = Math.Clamp(samples[i], -1f, 1f);
            pcm[i] = (short)(s * 32767f);
        }

        int dataSize = pcm.Length * 2;
        using var fs = new FileStream(path, FileMode.Create);
        using var bw = new BinaryWriter(fs);

        bw.Write(new[] { (byte)'R', (byte)'I', (byte)'F', (byte)'F' });
        bw.Write(36 + dataSize);
        bw.Write(new[] { (byte)'W', (byte)'A', (byte)'V', (byte)'E' });
        bw.Write(new[] { (byte)'f', (byte)'m', (byte)'t', (byte)' ' });
        bw.Write(16);
        bw.Write((short)1);
        bw.Write((short)1);
        bw.Write(sampleRate);
        bw.Write(sampleRate * 2);
        bw.Write((short)2);
        bw.Write((short)16);
        bw.Write(new[] { (byte)'d', (byte)'a', (byte)'t', (byte)'a' });
        bw.Write(dataSize);

        foreach (var s in pcm) bw.Write(s);
    }
}
