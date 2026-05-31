using System;
using System.IO;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace Eve;

class Program
{
    static string GetOutputDir()
    {
        var baseDir = AppDomain.CurrentDomain.BaseDirectory;
        var outputDir = Path.Combine(baseDir, "..", "..", "..", "output");
        Directory.CreateDirectory(outputDir);
        return Path.GetFullPath(outputDir);
    }

    static void Main(string[] args)
    {
        if (args.Length > 0 && args[0] == "--test-native")
        {
            TestNative();
            return;
        }

        if (args.Length > 0 && args[0] == "--generate")
        {
            var count = args.Length > 1 ? int.Parse(args[1]) : 50;
            var cfg = new Utils.Config();
            using var gen = new Data.SyntheticDataGenerator();
            gen.Generate(cfg, count);
            return;
        }

        if (args.Length > 0 && args[0] == "--generate-audio")
        {
            var numTokens = args.Length > 1 ? int.Parse(args[1]) : 1500;
            var temperature = args.Length > 2 ? float.Parse(args[2]) : 0.9f;
            var promptPath = args.Length > 3 ? args[3] : null;
            var outputPath = args.Length > 4 ? args[4] : null;
            GenerateAudio(numTokens, temperature, promptPath, outputPath);
            return;
        }

        var resume = args.Length > 0 && args[0] == "--resume";
        TrainFullModel(resume);
    }

    static void TrainFullModel(bool resume)
    {
        var cfg = new Utils.Config();
        Console.WriteLine("=== Full Model Training (VQ-VAE + Transformer) ===\n");

        // Phase 0: Pre-encode all audio to codes
        Console.WriteLine("Phase 0: Encoding all audio to codes...");
        var dataset = new Data.AudioDataset(cfg.DataPath, cfg.AudioLengthSamples);
        if (dataset.Count == 0)
        {
            Console.WriteLine($"\u26a0 No training data in '{cfg.DataPath}/'. Generate with: dotnet run -- --generate 100");
            return;
        }
        Console.WriteLine($"Loaded {dataset.Count} audio files.\n");

        // Create handle for encoding
        using var encHandle = new Native.EveNativeHandle(
            codebookSize: 256, latentDim: 64, embedDim: 64,
            numHeads: 4, numLayers: 4, feedForwardDim: 256,
            maxSeqLen: 2048, sampleRate: 24000, audioLen: 24000);
        encHandle.InitWeights(42);
        encHandle.SetCommitmentCost(1.0f);
        encHandle.SetEmaDecay(0.99f);

        // Encode all files to codes and save
        string codesPath = Path.Combine(cfg.DataPath, "codes.bin");
        Console.WriteLine($"Encoding {dataset.Count} files -> {codesPath}");
        var sw = System.Diagnostics.Stopwatch.StartNew();
        int totalCodes = 0;
        int[] codeBuffer = new int[2048];

        using (var fs = new FileStream(codesPath, FileMode.Create))
        using (var bw = new BinaryWriter(fs))
        {
            for (int i = 0; i < dataset.Count; i++)
            {
                float[] audio = dataset.Get(i);
                int numCodes = encHandle.Encode(audio, 1, 24000, codeBuffer, codeBuffer.Length);
                if (numCodes > 0)
                {
                    bw.Write(numCodes);
                    for (int j = 0; j < numCodes; j++)
                        bw.Write(codeBuffer[j]);
                    totalCodes += numCodes;
                }
                if ((i + 1) % 50 == 0)
                    Console.WriteLine($"  Encoded {(i + 1)}/{dataset.Count} files...");
            }
        }
        sw.Stop();
        Console.WriteLine($"Encoded {totalCodes} total codes from {dataset.Count} files in {sw.Elapsed.TotalSeconds:F1}s\n");

        // Phase 1: Train VQ-VAE
        Console.WriteLine("=== Phase 1: Train VQ-VAE ===\n");
        Console.WriteLine("Creating VQ-VAE handle...");
        using var vqvae = new Native.EveNativeHandle(
            codebookSize: 256, latentDim: 64, embedDim: 64,
            numHeads: 4, numLayers: 4, feedForwardDim: 256,
            maxSeqLen: 2048, sampleRate: 24000, audioLen: 24000);

        if (resume)
        {
            var vqvaePath = Path.Combine(GetOutputDir(), "eve_vqvae_trained.bin");
            if (File.Exists(vqvaePath))
            {
                vqvae.LoadWeights(vqvaePath);
                Console.WriteLine($"Loaded existing VQ-VAE weights from {vqvaePath}\n");
            }
            else
            {
                Console.WriteLine("No existing VQ-VAE weights found, training from scratch.\n");
                vqvae.InitWeights(42);
            }
        }
        else
        {
            vqvae.InitWeights(42);
        }

        vqvae.SetLearningRate(1e-4f);
        vqvae.SetCommitmentCost(1.0f);
        vqvae.SetEmaDecay(0.99f);

        int vqvaeSteps = 5000;
        float[] audioBuffer = new float[24000];
        sw = System.Diagnostics.Stopwatch.StartNew();

        for (int step = 0; step < vqvaeSteps; step++)
        {
            int fileIdx = Random.Shared.Next(dataset.Count);
            float[] audio = dataset.Get(fileIdx);
            Array.Copy(audio, audioBuffer, 24000);

            float loss = vqvae.TrainVqVaeStep(audioBuffer, 1, 24000);

            if (step % 500 == 0 || step == vqvaeSteps - 1)
            {
                double elapsed = sw.Elapsed.TotalSeconds;
                Console.WriteLine($"  VQ-VAE Step {step,4}/{vqvaeSteps}  loss={loss:F6}  ({elapsed / (step + 1) * 1000:F1} ms/step)");
            }
        }
        sw.Stop();
        Console.WriteLine($"\nVQ-VAE training done: {sw.Elapsed.TotalSeconds:F1}s\n");

        // Save VQ-VAE weights
        string outputDir = GetOutputDir();
        string vqvaeSavePath = Path.Combine(outputDir, "eve_vqvae_trained.bin");
        vqvae.SaveWeights(vqvaeSavePath);
        Console.WriteLine($"Saved VQ-VAE weights to {vqvaeSavePath}\n");

        // Phase 2: Train Transformer
        Console.WriteLine("=== Phase 2: Train Transformer ===\n");
        Console.WriteLine("Creating Transformer handle...");
        using var transformer = new Native.EveNativeHandle(
            codebookSize: 256, latentDim: 64, embedDim: 512,
            numHeads: 8, numLayers: 6, feedForwardDim: 2048,
            maxSeqLen: 2048, sampleRate: 24000, audioLen: 24000);

        if (resume)
        {
            var transPath = Path.Combine(outputDir, "eve_transformer_trained.bin");
            if (File.Exists(transPath))
            {
                transformer.LoadWeights(transPath);
                Console.WriteLine($"Loaded existing Transformer weights from {transPath}\n");
            }
            else
            {
                Console.WriteLine("No existing Transformer weights found, training from scratch.\n");
                transformer.InitWeights(42);
            }
        }
        else
        {
            transformer.InitWeights(42);
        }

        transformer.SetLearningRate(3e-4f);
        Console.WriteLine("Initialized Transformer weights.\n");

        // Load codes and train
        Console.WriteLine($"Loading codes from {codesPath}...");
        List<int> allCodesList = new();
        List<int> seqLensList = new();
        using (var fs = new FileStream(codesPath, FileMode.Open))
        using (var br = new BinaryReader(fs))
        {
            while (br.BaseStream.Position < br.BaseStream.Length)
            {
                int len = br.ReadInt32();
                if (len <= 0 || len > 2048) break;
                seqLensList.Add(len);
                for (int j = 0; j < len; j++)
                    allCodesList.Add(br.ReadInt32());
            }
        }
        int[] allCodes = allCodesList.ToArray();
        int[] allSeqLens = seqLensList.ToArray();
        Console.WriteLine($"Loaded {allCodes.Length} total codes across {allSeqLens.Length} sequences.\n");

        int transSteps = 10000;
        int batchSize = 1;
        sw = System.Diagnostics.Stopwatch.StartNew();
        int codeOffset = 0;

        for (int step = 0; step < transSteps; step++)
        {
            // Pick a random sequence
            int seqIdx = Random.Shared.Next(allSeqLens.Length);
            int seqLen = allSeqLens[seqIdx];
            if (seqLen < 32) { step--; continue; }  // Skip too-short sequences

            int[] seqCodes = new int[seqLen];
            Array.Copy(allCodes, codeOffset, seqCodes, 0, seqLen);
            codeOffset = (codeOffset / 4 + 1) * 4;  // Move to next sequence

            float loss = transformer.TrainTransformerStep(seqCodes, batchSize, seqLen);

            if (step % 500 == 0 || step == transSteps - 1)
            {
                double elapsed = sw.Elapsed.TotalSeconds;
                Console.WriteLine($"  Transformer Step {step,4}/{transSteps}  loss={loss:F6}  ({elapsed / (step + 1) * 1000:F1} ms/step)");
            }
        }
        sw.Stop();
        Console.WriteLine($"\nTransformer training done: {sw.Elapsed.TotalSeconds:F1}s\n");

        // Save Transformer weights
        string transSavePath = Path.Combine(outputDir, "eve_transformer_trained.bin");
        transformer.SaveWeights(transSavePath);
        Console.WriteLine($"Saved Transformer weights to {transSavePath}");
        Console.WriteLine("\n=== Full training pipeline complete! ===");
    }

    static void GenerateAudio(int numTokens, float temperature, string? promptPath, string? outputPath)
    {
        Console.WriteLine("=== Audio Generation ===\n");
        Console.WriteLine($"Tokens: {numTokens}, Temperature: {temperature}");

        // Load trained weights
        string outputDir = GetOutputDir();
        string vqvaePath = Path.Combine(outputDir, "eve_vqvae_trained.bin");
        string transPath = Path.Combine(outputDir, "eve_transformer_trained.bin");

        if (!File.Exists(vqvaePath) || !File.Exists(transPath))
        {
            Console.WriteLine("Error: trained weights not found. Run training first with: dotnet run");
            return;
        }

        // Create VQ-VAE handle for encoding prompt and decoding output
        using var vqvae = new Native.EveNativeHandle(
            codebookSize: 256, latentDim: 64, embedDim: 64,
            numHeads: 4, numLayers: 4, feedForwardDim: 256,
            maxSeqLen: 2048, sampleRate: 24000, audioLen: 24000);
        vqvae.LoadWeights(vqvaePath);

        // Create Transformer handle for generation
        using var transformer = new Native.EveNativeHandle(
            codebookSize: 256, latentDim: 64, embedDim: 512,
            numHeads: 8, numLayers: 6, feedForwardDim: 2048,
            maxSeqLen: 2048, sampleRate: 24000, audioLen: 24000);
        transformer.LoadWeights(transPath);

        float[]? promptAudio = null;
        if (promptPath != null && File.Exists(promptPath))
        {
            Console.WriteLine($"Loading prompt audio from {promptPath}...");
            promptAudio = Utils.AudioIO.ReadWav(promptPath, out int _);
            Console.WriteLine($"Prompt audio: {promptAudio.Length} samples\n");
        }
        else
        {
            Console.WriteLine("No prompt provided, generating from scratch.\n");
        }

        // Generate audio
        int maxOutputSamples = 24000 * 10;  // Up to 10 seconds
        float[] outputAudio = new float[maxOutputSamples];
        int promptLen = promptAudio != null ? promptAudio.Length : 0;

        Console.WriteLine("Generating...");
        int actualLen = transformer.Generate(
            promptAudio ?? Array.Empty<float>(),
            promptLen,
            numTokens,
            temperature,
            outputAudio,
            outputAudio.Length);

        // Write output WAV file
        string outPath = outputPath ?? Path.Combine(outputDir, "generated_output.wav");
        Utils.AudioIO.WriteWav(outPath, outputAudio, 24000);
        Console.WriteLine($"\nGenerated {actualLen} samples ({actualLen / 24000.0:F2}s) -> {outPath}");
    }

    static void TestNative()
    {
        Console.WriteLine("Testing Eve.Native P/Invoke wrapper...\n");

        using var eve = new Native.EveNativeHandle(
            codebookSize: 1024,
            latentDim: 64,
            embedDim: 512,
            numHeads: 8,
            numLayers: 6,
            feedForwardDim: 2048,
            maxSeqLen: 8192,
            sampleRate: 24000,
            audioLen: 120000
        );
        Console.WriteLine("1. Created handle.");

        eve.InitWeights(42);
        Console.WriteLine("2. Initialized weights.");

        float[] audio = new float[120000];
        for (int i = 0; i < audio.Length; i++)
            audio[i] = 0.5f * MathF.Sin(2 * MathF.PI * 440.0f * i / 24000.0f);
        Console.WriteLine("3. Created sine wave audio.");

        float[] z = new float[120000 * 64];
        int zLen = eve.EncoderForward(audio, 1, audio.Length, z, z.Length);
        float[] zOrig = (float[])z.Clone();
        Console.WriteLine($"4. Encoder forward: seq_len={zLen}, z[0..4]={z[0]:F6} {z[1]:F6} {z[2]:F6} {z[3]:F6} {z[4]:F6}");

        int[] codes = new int[zLen];
        float[] qz = new float[z.Length];
        eve.Quantize(z, 1, zLen, codes, qz, qz.Length);
        Console.WriteLine($"5. Quantize: codes[0..4]={codes[0]} {codes[1]} {codes[2]} {codes[3]} {codes[4]}");

        int[] codes2 = new int[10000];
        int n2 = eve.Encode(audio, 1, audio.Length, codes2, codes2.Length);
        Console.WriteLine($"6. Eve.Encode: {n2} codes, first 5={codes2[0]} {codes2[1]} {codes2[2]} {codes2[3]} {codes2[4]}");

        float[] recon = new float[120000];
        eve.DecoderForward(qz, 1, zLen, recon, recon.Length);
        Console.WriteLine($"7. Decoder forward: recon[0..4]={recon[0]:F6} {recon[1]:F6} {recon[2]:F6} {recon[3]:F6} {recon[4]:F6}");

        string testOutputDir = GetOutputDir();
        string csharpWeightsPath = Path.Combine(testOutputDir, "eve_csharp_weights.bin");
        eve.SaveWeights(csharpWeightsPath);
        Console.WriteLine($"8. Saved weights to {csharpWeightsPath}.");

        using var eve2 = new Native.EveNativeHandle(1024, 64, 512, 8, 6, 2048, 8192, 24000, 120000);
        eve2.LoadWeights(csharpWeightsPath);
        Console.WriteLine("9. Loaded weights into new handle.");

        float[] z2 = new float[120000 * 64];
        int zLen2 = eve2.EncoderForward(audio, 1, audio.Length, z2, z2.Length);
        Console.WriteLine($"10. Reloaded encoder: z[0..4]={z2[0]:F6} {z2[1]:F6} {z2[2]:F6} {z2[3]:F6} {z2[4]:F6}");

        float diff = Math.Abs(zOrig[0] - z2[0]);
        Console.WriteLine($"    diff={diff:E}, zOrig[0]={zOrig[0]:E}, z2[0]={z2[0]:E}");

        int mismatches = 0;
        for (int i = 0; i < zLen && i < zLen2; i++)
            if (Math.Abs(zOrig[i] - z2[i]) > 1e-6f) mismatches++;
        Console.WriteLine($"    Total mismatches (eps=1e-6): {mismatches} / {zLen}");

        Console.WriteLine("\nAll C# P/Invoke tests passed!");
    }
}
