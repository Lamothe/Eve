using System;
using System.Collections.Generic;
using System.IO;
using Eve.Utils;

namespace Eve.Data;

public class AudioDataset
{
    private readonly List<float[]> _cachedData = new();
    private readonly int _targetLen;

    public int Count => _cachedData.Count;

    public AudioDataset(string dirPath, int targetLen)
    {
        _targetLen = targetLen;
        if (!Directory.Exists(dirPath)) return;

        var files = Directory.GetFiles(dirPath, "*.wav", SearchOption.AllDirectories);
        if (files.Length == 0) return;

        Console.WriteLine($"Loading {files.Length} audio files into RAM...");

        foreach (var f in files)
        {
            try
            {
                var samples = AudioIO.ReadWav(f, out int sr);
                
                // Crop or zero-pad to exactly targetLen (24,000 samples)
                var processed = new float[targetLen];
                if (samples.Length >= targetLen)
                {
                    Array.Copy(samples, processed, targetLen);
                }
                else
                {
                    Array.Copy(samples, processed, samples.Length);
                }
                
                _cachedData.Add(processed);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Warning: Failed to load {f}: {ex.Message}");
            }
        }
    }

    public float[] Get(int idx)
    {
        if (idx < 0 || idx >= Count) throw new IndexOutOfRangeException();
        return _cachedData[idx];
    }

    public float[] GetBatch(int batchSize)
    {
        if (Count == 0) throw new InvalidOperationException("No audio files loaded");

        float[] batch = new float[batchSize * _targetLen];
        for (int b = 0; b < batchSize; b++)
        {
            int idx = Random.Shared.Next(Count);
            Array.Copy(_cachedData[idx], 0, batch, b * _targetLen, _targetLen);
        }
        return batch;
    }
}
