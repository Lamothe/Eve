using System;
using System.Numerics;

namespace Eve.Utils;

/// <summary>
/// Short-Time Fourier Transform utilities for audio processing.
/// </summary>
public static class STFT
{
    public const int DefaultWindowSize = 1024;
    public const int DefaultHopSize = 256;
    public const int DefaultFFTSize = 1024;

    /// <summary>
    /// Compute Short-Time Fourier Transform.
    /// Returns complex spectrogram [frames, bins] where bins = fftSize/2 + 1.
    /// </summary>
    public static Complex[,] Transform(float[] audio, int windowSize = DefaultWindowSize, 
        int hopSize = DefaultHopSize, int fftSize = DefaultFFTSize)
    {
        if (fftSize < windowSize)
            throw new ArgumentException("FFT size must be >= window size");

        int numFrames = (audio.Length - windowSize) / hopSize + 1;
        int numBins = fftSize / 2 + 1;
        var spectrogram = new Complex[numFrames, numBins];
        var window = HannWindow(windowSize);

        for (int frame = 0; frame < numFrames; frame++)
        {
            int start = frame * hopSize;
            var frameData = new float[fftSize];

            // Apply window
            for (int i = 0; i < windowSize; i++)
                frameData[i] = audio[start + i] * window[i];

            // Zero-pad if fftSize > windowSize
            // (already zero from array initialization)

            // Compute FFT
            var spectrum = FFT(frameData);

            // Store positive frequencies only
            for (int bin = 0; bin < numBins; bin++)
                spectrogram[frame, bin] = spectrum[bin];
        }

        return spectrogram;
    }

    /// <summary>
    /// Inverse STFT: reconstruct audio from complex spectrogram.
    /// </summary>
    public static float[] Inverse(Complex[,] spectrogram, int windowSize = DefaultWindowSize,
        int hopSize = DefaultHopSize, int fftSize = DefaultFFTSize)
    {
        int numFrames = spectrogram.GetLength(0);
        int numBins = spectrogram.GetLength(1);
        int audioLength = (numFrames - 1) * hopSize + windowSize;
        var audio = new float[audioLength];
        var window = HannWindow(windowSize);
        var windowSum = new float[audioLength];

        for (int frame = 0; frame < numFrames; frame++)
        {
            int start = frame * hopSize;

            // Reconstruct full spectrum (mirror for negative frequencies)
            var spectrum = new Complex[fftSize];
            for (int bin = 0; bin < numBins; bin++)
                spectrum[bin] = spectrogram[frame, bin];
            for (int bin = numBins; bin < fftSize; bin++)
                spectrum[bin] = Complex.Conjugate(spectrum[fftSize - bin]);

            // Inverse FFT
            var frameData = IFFT(spectrum);

            // Apply window and overlap-add
            for (int i = 0; i < windowSize; i++)
            {
                audio[start + i] += (float)frameData[i].Real * window[i];
                windowSum[start + i] += window[i] * window[i];
            }
        }

        // Normalize by window sum (avoid division by zero)
        for (int i = 0; i < audioLength; i++)
        {
            if (windowSum[i] > 1e-8f)
                audio[i] /= windowSum[i];
        }

        return audio;
    }

    /// <summary>
    /// Generate Hann window.
    /// </summary>
    public static float[] HannWindow(int size)
    {
        var window = new float[size];
        for (int i = 0; i < size; i++)
            window[i] = 0.5f * (1.0f - MathF.Cos(2.0f * MathF.PI * i / (size - 1)));
        return window;
    }

    /// <summary>
    /// Compute FFT using Cooley-Tukey algorithm.
    /// Input length must be a power of 2.
    /// </summary>
    public static Complex[] FFT(float[] input)
    {
        int n = input.Length;
        if ((n & (n - 1)) != 0)
            throw new ArgumentException("Input length must be a power of 2");

        var x = new Complex[n];
        for (int i = 0; i < n; i++)
            x[i] = new Complex(input[i], 0);

        return FFT(x);
    }

    /// <summary>
    /// Compute FFT of complex input.
    /// </summary>
    public static Complex[] FFT(Complex[] x)
    {
        int n = x.Length;
        if (n == 1)
            return new[] { x[0] };

        // Bit-reversal permutation
        var y = new Complex[n];
        int bits = (int)Math.Log2(n);
        for (int i = 0; i < n; i++)
        {
            int j = BitReverse(i, bits);
            y[j] = x[i];
        }

        // Cooley-Tukey iterative FFT
        for (int len = 2; len <= n; len *= 2)
        {
            double angle = -2.0 * Math.PI / len;
            var wlen = new Complex(Math.Cos(angle), Math.Sin(angle));

            for (int i = 0; i < n; i += len)
            {
                var w = Complex.One;
                for (int j = 0; j < len / 2; j++)
                {
                    var u = y[i + j];
                    var v = y[i + j + len / 2] * w;
                    y[i + j] = u + v;
                    y[i + j + len / 2] = u - v;
                    w *= wlen;
                }
            }
        }

        return y;
    }

    /// <summary>
    /// Compute inverse FFT.
    /// </summary>
    public static Complex[] IFFT(Complex[] x)
    {
        int n = x.Length;
        var conjugate = new Complex[n];
        for (int i = 0; i < n; i++)
            conjugate[i] = Complex.Conjugate(x[i]);

        var result = FFT(conjugate);

        for (int i = 0; i < n; i++)
            result[i] = Complex.Conjugate(result[i]) / n;

        return result;
    }

    private static int BitReverse(int x, int bits)
    {
        int result = 0;
        for (int i = 0; i < bits; i++)
        {
            result = (result << 1) | (x & 1);
            x >>= 1;
        }
        return result;
    }
}
