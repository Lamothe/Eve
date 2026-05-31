using KokoroSharp;
using KokoroSharp.Core;
using KokoroSharp.Processing;
using static TorchSharp.torch;

namespace Eve.Data;

public class SyntheticDataGenerator : IDisposable
{
    private readonly KokoroTTS _tts;
    private readonly KokoroVoice[] _voices;
    private static readonly string[] Prompts =
    [
        "Hello, how are you today?",
        "The quick brown fox jumps over the lazy dog.",
        "She sells seashells by the seashore.",
        "How much wood would a woodchuck chuck?",
        "I am excited to start this new project.",
        "Please follow the instructions carefully.",
        "What is the meaning of life?",
        "The rain in Spain falls mainly on the plain.",
        "This is a test of the emergency broadcast system.",
        "I love listening to music while I work.",
        "Can you hear me clearly?",
        "Stop right there and listen.",
        "Good morning, it is a beautiful day.",
        "The future of artificial intelligence is bright.",
        "One small step for man, one giant leap for mankind.",
        "To be or not to be, that is the question.",
        "I am so happy to see you today.",
        "This is absolutely terrible and unacceptable.",
        "The sixth sick sheik's sixth sheep's sick.",
        "Unique New York, New York's unique.",
        "Around the rugged rocks the ragged rascal ran.",
        "Peter Piper picked a peck of pickled peppers.",
        "How many cookies could a good cook cook?",
        "I saw Susie sitting in a shoe shine shop.",
        "Fuzzy Wuzzy was a bear, Fuzzy Wuzzy had no hair.",
        "The five boxing wizards jump quickly.",
        "Pack my box with five dozen liquor jugs.",
        "The jay, pig, fox, zebra and my wolves quack.",
        "In the beginning there was nothing but silence.",
        "All human beings are born free and equal in dignity.",
        "The only limit to our realization of tomorrow is our doubts.",
        "Science is a way of thinking much more than a body of knowledge.",
        "Music is the universal language of mankind.",
        "The best time to plant a tree was twenty years ago.",
        "I have a dream that one day this nation will rise up.",
        "Ask not what your country can do for you.",
        "That's one small step for man, one giant leap for mankind.",
        "Four score and seven years ago our fathers brought forth.",
        "The only thing we have to fear is fear itself.",
        "In the middle of difficulty lies opportunity.",
    ];

    public SyntheticDataGenerator()
    {
        Console.WriteLine("Loading Kokoro TTS model (first run downloads ~320MB)...");
        _tts = KokoroTTS.LoadModel();
        _voices = [.. KokoroVoiceManager.GetVoices(KokoroLanguage.AmericanEnglish, KokoroGender.Female)];
        Console.WriteLine($"Loaded {_voices.Length} voices");
    }

    public void Generate(Utils.Config cfg, int count)
    {
        Directory.CreateDirectory(cfg.DataPath);
        var rng = new Random(42);
        var existing = Directory.GetFiles(cfg.DataPath, "*.wav").Length;
        if (existing > 0)
        {
            Console.Write($"Data directory has {existing} files. Overwrite? (y/N): ");
            var response = Console.ReadLine()?.Trim().ToLower();
            if (response != "y" && response != "yes")
            {
                Console.WriteLine("Skipping generation.");
                return;
            }
        }

        for (int i = 0; i < count; i++)
        {
            var text = Prompts[rng.Next(Prompts.Length)];
            var voice = _voices[rng.Next(_voices.Length)];
            var speed = 0.9f + (float)rng.NextDouble() * 0.3f;

            Console.Write($"Generating {i + 1}/{count}: \"{Truncate(text, 50)}\" [{voice.Name}] [{speed:F1}x]... ");
            var samples = Synthesize(text, voice, speed);

            if (samples.Length == 0)
            {
                Console.WriteLine("empty output, skipping");
                continue;
            }

            using (no_grad())
            {
                var t = tensor(samples, float32).reshape(1, -1);
                var path = Path.Combine(cfg.DataPath, $"synth_{i:D4}_{voice.Name}_{speed:F1}x.wav");
                Utils.AudioIO.WriteWav(path, t, cfg.SampleRate);
            }

            Console.WriteLine($"{samples.Length / (double)cfg.SampleRate:F1}s");
        }

        Console.WriteLine($"\nGenerated {count} files in '{cfg.DataPath}/'");
    }

    private float[] Synthesize(string text, KokoroVoice voice, float speed)
    {
        var tokens = Tokenizer.Tokenize(text.Trim());
        if (tokens.Length == 0) return [];

        var result = new List<float>();
        var segments = new List<int[]> { tokens };
        float[,,] voiceStyle = voice;
        var job = KokoroJob.Create(segments, voiceStyle, speed, samples => result.AddRange(samples));

        _tts.EnqueueJob(job);
        while (!job.isDone) Thread.Sleep(5);

        return [.. result];
    }

    private static string Truncate(string s, int max) => s.Length <= max ? s : s[..max] + "...";

    public void Dispose()
    {
        _tts?.Dispose();
    }
}
