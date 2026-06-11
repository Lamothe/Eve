using System;
using System.Linq;
using System.Numerics;
using Eve.Utils;

namespace Eve.Data;

/// <summary>
/// Tokenizes audio using STFT and top-K frequency bin selection.
/// Each frame produces K tokens (bin indices 0 to numBins-1).
/// </summary>
public class AudioTokenizer
{
    public int WindowSize { get; }
    public int HopSize { get; }
    public int FFTSize { get; }
    public int NumBins { get; }
    public int TopK { get; }
    public int SampleRate { get; }

    /// <summary>
    /// Number of frames per second of audio.
    /// </summary>
    public int FramesPerSecond => SampleRate / HopSize;

    public AudioTokenizer(int sampleRate = 24000, int windowSize = 1024, 
        int hopSize = 256, int fftSize = 1024, int topK = 32)
    {
        SampleRate = sampleRate;
        WindowSize = windowSize;
        HopSize = hopSize;
        FFTSize = fftSize;
        NumBins = fftSize / 2 + 1;
        TopK = topK;
    }

    /// <summary>
    /// Encode audio to token sequence.
    /// Returns flat array of bin indices: [frame0_bin0, frame0_bin1, ..., frame0_binK-1, frame1_bin0, ...].
    /// </summary>
    public int[] Encode(float[] audio)
    {
        var spectrogram = STFT.Transform(audio, WindowSize, HopSize, FFTSize);
        int numFrames = spectrogram.GetLength(0);
        var tokens = new int[numFrames * TopK];

        for (int frame = 0; frame < numFrames; frame++)
        {
            // Get magnitudes for this frame
            var magnitudes = new (int bin, float mag)[NumBins];
            for (int bin = 0; bin < NumBins; bin++)
                magnitudes[bin] = (bin, (float)spectrogram[frame, bin].Magnitude);

            // Sort by magnitude descending, take top K
            var topBins = magnitudes
                .OrderByDescending(x => x.mag)
                .Take(TopK)
                .Select(x => x.bin)
                .OrderBy(x => x)  // Sort bins in ascending order for consistency
                .ToArray();

            // Store tokens
            for (int k = 0; k < TopK; k++)
                tokens[frame * TopK + k] = topBins[k];
        }

        return tokens;
    }

    /// <summary>
    /// Decode token sequence back to audio.
    /// Reconstructs STFT with unit magnitude at specified bins, then applies iSTFT.
    /// </summary>
    public float[] Decode(int[] tokens, int numFrames)
    {
        if (tokens.Length != numFrames * TopK)
            throw new ArgumentException($"Expected {numFrames * TopK} tokens, got {tokens.Length}");

        var spectrogram = new Complex[numFrames, NumBins];

        for (int frame = 0; frame < numFrames; frame++)
        {
            // Set unit magnitude at specified bins
            for (int k = 0; k < TopK; k++)
            {
                int bin = tokens[frame * TopK + k];
                if (bin >= 0 && bin < NumBins)
                    spectrogram[frame, bin] = Complex.One;
            }
        }

        return STFT.Inverse(spectrogram, WindowSize, HopSize, FFTSize);
    }

    /// <summary>
    /// Decode token sequence back to audio, preserving original magnitudes.
    /// Requires the original spectrogram to extract magnitudes.
    /// </summary>
    public float[] DecodeWithMagnitudes(int[] tokens, Complex[,] originalSpectrogram, int numFrames)
    {
        if (tokens.Length != numFrames * TopK)
            throw new ArgumentException($"Expected {numFrames * TopK} tokens, got {tokens.Length}");

        var spectrogram = new Complex[numFrames, NumBins];

        for (int frame = 0; frame < numFrames; frame++)
        {
            for (int k = 0; k < TopK; k++)
            {
                int bin = tokens[frame * TopK + k];
                if (bin >= 0 && bin < NumBins && frame < originalSpectrogram.GetLength(0))
                {
                    // Use original magnitude and phase
                    spectrogram[frame, bin] = originalSpectrogram[frame, bin];
                }
            }
        }

        return STFT.Inverse(spectrogram, WindowSize, HopSize, FFTSize);
    }

    /// <summary>
    /// Get the number of frames for a given audio length.
    /// </summary>
    public int GetNumFrames(int audioLength)
    {
        return (audioLength - WindowSize) / HopSize + 1;
    }

    /// <summary>
    /// Get the audio length for a given number of frames.
    /// </summary>
    public int GetAudioLength(int numFrames)
    {
        return (numFrames - 1) * HopSize + WindowSize;
    }
}
