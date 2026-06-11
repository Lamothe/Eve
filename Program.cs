using System;
using System.IO;
using System.Linq;
using Eve.Data;
using Eve.Native;
using Eve.Utils;

namespace Eve
{
    class Program
    {
        static void Main(string[] args)
        {
            if (args.Length == 0)
            {
                Console.WriteLine("Usage:");
                Console.WriteLine("  eve train <audio_dir> <output_model> [--resume <checkpoint>]");
                Console.WriteLine("  eve generate <model> <prompt_audio> <output_audio> [num_frames] [temperature]");
                return;
            }

            string command = args[0].ToLower();

            switch (command)
            {
                case "train":
                    if (args.Length < 3)
                    {
                        Console.WriteLine("Usage: eve train <audio_dir> <output_model> [--resume <checkpoint>]");
                        return;
                    }
                    string? checkpointPath = null;
                    for (int i = 3; i < args.Length - 1; i++)
                    {
                        if (args[i] == "--resume")
                        {
                            checkpointPath = args[i + 1];
                        }
                    }
                    Train(args[1], args[2], checkpointPath);
                    break;

                case "generate":
                    if (args.Length < 4)
                    {
                        Console.WriteLine("Usage: eve generate <model> <prompt_audio> <output_audio> [num_frames] [temperature]");
                        return;
                    }
                    int numFrames = args.Length > 4 ? int.Parse(args[4]) : 100;
                    float temperature = args.Length > 5 ? float.Parse(args[5]) : 0.8f;
                    Generate(args[1], args[2], args[3], numFrames, temperature);
                    break;

                default:
                    Console.WriteLine($"Unknown command: {command}");
                    break;
            }
        }

        static void Train(string audioDir, string outputPath, string? checkpointPath)
        {
            Console.WriteLine($"Training on audio from: {audioDir}");

            // Configuration
            int sampleRate = 24000;
            int windowSize = 1024;
            int hopSize = 256;
            int fftSize = 1024;
            int topK = 32;
            int vocabSize = fftSize / 2 + 1; // 513 bins

            int embedDim = 256;
            int numHeads = 4;
            int numLayers = 4;
            int feedForwardDim = 1024;
            int maxSeqLen = 128;

            int numEpochs = 50;
            int checkpointEvery = 5; // Save checkpoint every N epochs
            float learningRate = 0.00001f;

            // Initialize tokenizer
            var tokenizer = new AudioTokenizer(sampleRate, windowSize, hopSize, fftSize, topK);

            // Load and tokenize audio files
            Console.WriteLine("Loading and tokenizing audio files...");
            var audioFiles = Directory.GetFiles(audioDir, "*.wav");
            var allTokens = new System.Collections.Generic.List<int[]>();

            foreach (var file in audioFiles)
            {
                try
                {
                    var audio = AudioIO.LoadWav(file);
                    if (audio.SampleRate != sampleRate)
                    {
                        continue;
                    }

                    var tokens = tokenizer.Encode(audio.Samples);
                    allTokens.Add(tokens);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"  Error loading {Path.GetFileName(file)}: {ex.Message}");
                }
            }

            if (allTokens.Count == 0)
            {
                Console.WriteLine("No audio files loaded successfully");
                return;
            }

            Console.WriteLine($"Loaded {allTokens.Count} files, total {allTokens.Sum(t => t.Length)} tokens");
            Console.WriteLine("Initializing model...");
            Console.Out.Flush();
            using var model = new EveNativeHandle(vocabSize, embedDim, numHeads, numLayers,
                                                   feedForwardDim, maxSeqLen);
            model.SetLearningRate(learningRate);

            int startEpoch = 0;
            if (checkpointPath != null && File.Exists(checkpointPath))
            {
                Console.WriteLine($"Loading checkpoint from {checkpointPath}...");
                int iter = model.LoadCheckpoint(checkpointPath);
                if (iter >= 0)
                {
                    Console.WriteLine($"Checkpoint loaded (AdamW iter={iter})");
                }
                else
                {
                    Console.WriteLine($"Failed to load checkpoint (error {iter}), initializing fresh");
                    model.InitWeights(42);
                }
            }
            else
            {
                model.InitWeights(42);
            }

            // Training loop
            Console.WriteLine($"Training for {numEpochs} epochs (starting from epoch {startEpoch + 1})...");
            Console.Out.Flush();
            int globalStep = 0;

            for (int epoch = startEpoch; epoch < numEpochs; epoch++)
            {
                Console.WriteLine($"Epoch {epoch + 1}/{numEpochs} starting...");
                Console.Out.Flush();
                float epochLoss = 0.0f;
                int epochSteps = 0;

                // Shuffle files
                var shuffled = allTokens.OrderBy(x => Guid.NewGuid()).ToList();

                foreach (var tokens in shuffled)
                {
                    // Split into chunks of maxSeqLen
                    for (int start = 0; start < tokens.Length - maxSeqLen; start += maxSeqLen / 2)
                    {
                        int chunkLen = Math.Min(maxSeqLen, tokens.Length - start);
                        if (chunkLen < 2) continue;

                        var chunk = new int[chunkLen];
                        Array.Copy(tokens, start, chunk, 0, chunkLen);

                        float loss = model.TrainStep(chunk, chunkLen);
                        epochLoss += loss;
                        epochSteps++;
                        globalStep++;

                        if (globalStep % 10 == 0)
                        {
                            Console.WriteLine($"  Step {globalStep}: loss = {loss:F4}");
                        }
                    }
                }

                float avgLoss = epochSteps > 0 ? epochLoss / epochSteps : 0.0f;
                Console.WriteLine($"Epoch {epoch + 1}/{numEpochs}: avg loss = {avgLoss:F4}");

                // Save checkpoint periodically
                if ((epoch + 1) % checkpointEvery == 0)
                {
                    string ckptPath = $"{outputPath}.ckpt_ep{epoch + 1}";
                    int ckptResult = model.SaveCheckpoint(ckptPath);
                    if (ckptResult == 0)
                    {
                        Console.WriteLine($"Checkpoint saved to: {ckptPath}");
                    }
                    else
                    {
                        Console.WriteLine($"Failed to save checkpoint (error {ckptResult})");
                    }
                }
            }

            // Save final model
            int saveResult = model.SaveModel(outputPath);
            if (saveResult == 0)
            {
                Console.WriteLine($"Model saved to: {outputPath}");
            }
            else
            {
                Console.WriteLine($"Failed to save model (error {saveResult})");
            }
            Console.WriteLine("Training complete.");
        }

       static void Generate(string modelPath, string promptPath, string outputPath,
                            int numFrames, float temperature)
        {
            Console.WriteLine($"Generating from prompt: {promptPath}");

            // Configuration (must match training)
            int sampleRate = 24000;
            int windowSize = 1024;
            int hopSize = 256;
            int fftSize = 1024;
            int topK = 32;
            int vocabSize = fftSize / 2 + 1;

            int embedDim = 256;
            int numHeads = 4;
            int numLayers = 4;
            int feedForwardDim = 1024;
            int maxSeqLen = 128;

            // Initialize tokenizer
            var tokenizer = new AudioTokenizer(sampleRate, windowSize, hopSize, fftSize, topK);

            // Load prompt audio
            var promptAudio = AudioIO.LoadWav(promptPath);
            if (promptAudio.SampleRate != sampleRate)
            {
                Console.WriteLine($"Error: prompt sample rate {promptAudio.SampleRate} != {sampleRate}");
                return;
            }

            // Tokenize prompt
            var promptTokens = tokenizer.Encode(promptAudio.Samples);
            Console.WriteLine($"Prompt: {promptAudio.Samples.Length} samples -> {promptTokens.Length} tokens");

            // Initialize model
            Console.WriteLine("Loading model...");
            using var model = new EveNativeHandle(vocabSize, embedDim, numHeads, numLayers,
                                                    feedForwardDim, maxSeqLen);
            int loadResult = model.LoadModel(modelPath);
            if (loadResult == 0)
            {
                Console.WriteLine("Model loaded successfully");
            }
            else
            {
                Console.WriteLine($"Failed to load model (error {loadResult}), using random weights");
                model.InitWeights(42);
            }

            // Generate tokens
            Console.WriteLine($"Generating {numFrames} frames ({numFrames * topK} tokens)...");
            int numTokensToGenerate = numFrames * topK;
            var outputTokens = new int[numTokensToGenerate];

            // Use last maxSeqLen/2 tokens as prompt
            int promptLen = Math.Min(promptTokens.Length, maxSeqLen / 2);
            var prompt = new int[promptLen];
            Array.Copy(promptTokens, promptTokens.Length - promptLen, prompt, 0, promptLen);

            int generated = model.Generate(prompt, promptLen, outputTokens, numTokensToGenerate, temperature);
            Console.WriteLine($"Generated {generated} tokens");

            // Combine prompt and generated tokens
            var allTokens = new int[promptLen + generated];
            Array.Copy(prompt, 0, allTokens, 0, promptLen);
            Array.Copy(outputTokens, 0, allTokens, promptLen, generated);

            // Decode to audio
            Console.WriteLine("Decoding to audio...");
            int totalFrames = allTokens.Length / topK;
            var audio = tokenizer.Decode(allTokens, totalFrames);

            // Save output
            AudioIO.SaveWav(outputPath, audio, sampleRate);
            Console.WriteLine($"Saved {audio.Length} samples to {outputPath}");
        }
    }
}
