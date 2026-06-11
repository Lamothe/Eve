using System;
using System.IO;

namespace Eve.Utils
{
    public class AudioData
    {
        public float[] Samples { get; set; }
        public int SampleRate { get; set; }

        public AudioData(float[] samples, int sampleRate)
        {
            Samples = samples;
            SampleRate = sampleRate;
        }
    }

    public static class AudioIO
    {
        public static AudioData LoadWav(string path)
        {
            using var fs = File.OpenRead(path);
            using var br = new BinaryReader(fs);

            // Read RIFF header
            string riff = new string(br.ReadChars(4));
            if (riff != "RIFF")
                throw new Exception("Not a valid WAV file");

            br.ReadInt32(); // file size
            string wave = new string(br.ReadChars(4));
            if (wave != "WAVE")
                throw new Exception("Not a valid WAV file");

            // Read fmt chunk
            string fmt = new string(br.ReadChars(4));
            if (fmt != "fmt ")
                throw new Exception("Expected fmt chunk");

            int fmtSize = br.ReadInt32();
            short audioFormat = br.ReadInt16();
            short numChannels = br.ReadInt16();
            int sampleRate = br.ReadInt32();
            int byteRate = br.ReadInt32();
            short blockAlign = br.ReadInt16();
            short bitsPerSample = br.ReadInt16();

            if (audioFormat != 1)
                throw new Exception($"Unsupported audio format: {audioFormat} (only PCM supported)");

            if (numChannels != 1)
                throw new Exception($"Only mono audio supported (got {numChannels} channels)");

            // Skip extra fmt bytes if present
            if (fmtSize > 16)
                br.ReadBytes(fmtSize - 16);

            // Read data chunk
            string data = new string(br.ReadChars(4));
            if (data != "data")
                throw new Exception("Expected data chunk");

            int dataSize = br.ReadInt32();
            int numSamples = dataSize / (bitsPerSample / 8);

            float[] samples = new float[numSamples];

            if (bitsPerSample == 16)
            {
                for (int i = 0; i < numSamples; i++)
                {
                    short sample = br.ReadInt16();
                    samples[i] = sample / 32768.0f;
                }
            }
            else if (bitsPerSample == 24)
            {
                for (int i = 0; i < numSamples; i++)
                {
                    byte[] bytes = br.ReadBytes(3);
                    int sample = (bytes[2] << 16) | (bytes[1] << 8) | bytes[0];
                    if ((sample & 0x800000) != 0)
                        sample |= unchecked((int)0xFF000000);
                    samples[i] = sample / 8388608.0f;
                }
            }
            else if (bitsPerSample == 32)
            {
                for (int i = 0; i < numSamples; i++)
                {
                    int sample = br.ReadInt32();
                    samples[i] = sample / 2147483648.0f;
                }
            }
            else
            {
                throw new Exception($"Unsupported bit depth: {bitsPerSample}");
            }

            return new AudioData(samples, sampleRate);
        }

        public static void SaveWav(string path, float[] samples, int sampleRate)
        {
            using var fs = File.Create(path);
            using var bw = new BinaryWriter(fs);

            int bitsPerSample = 16;
            int numChannels = 1;
            int byteRate = sampleRate * numChannels * bitsPerSample / 8;
            short blockAlign = (short)(numChannels * bitsPerSample / 8);
            int dataSize = samples.Length * (bitsPerSample / 8);
            int fileSize = 36 + dataSize;

            // RIFF header
            bw.Write("RIFF".ToCharArray());
            bw.Write(fileSize);
            bw.Write("WAVE".ToCharArray());

            // fmt chunk
            bw.Write("fmt ".ToCharArray());
            bw.Write(16); // fmt chunk size
            bw.Write((short)1); // PCM
            bw.Write((short)numChannels);
            bw.Write(sampleRate);
            bw.Write(byteRate);
            bw.Write(blockAlign);
            bw.Write((short)bitsPerSample);

            // data chunk
            bw.Write("data".ToCharArray());
            bw.Write(dataSize);

            for (int i = 0; i < samples.Length; i++)
            {
                float clamped = Math.Max(-1.0f, Math.Min(1.0f, samples[i]));
                short sample = (short)(clamped * 32767.0f);
                bw.Write(sample);
            }
        }
    }
}
