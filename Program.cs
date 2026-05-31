using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace Eve;

class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("Eve - Audio-Native Transformer");
        Console.WriteLine($"TorchSharp: {torch.__version__}");
        Console.WriteLine($"CUDA: {torch.cuda.is_available()}");
        Console.WriteLine();

        var cfg = new Utils.Config();
        var device = torch.cuda.is_available() ? torch.CUDA : torch.CPU;
        Console.WriteLine($"Device: {device}");

        var vqvae = new Models.VQVAE(
            inChannels: 1,
            latentDim: cfg.LatentDim,
            codebookSize: cfg.CodebookSize,
            emaDecay: cfg.EmaDecay,
            emaEpsilon: cfg.EmaEpsilon,
            commitmentCost: cfg.CommitmentCost);

        var transformer = new Models.AudioTransformer(
            cfg.CodebookSize, cfg.EmbedDim, cfg.NumHeads, cfg.NumLayers, cfg.FeedForwardDim);

        Console.WriteLine($"VQ-VAE: {CountParams(vqvae):N0} params");
        Console.WriteLine($"Transformer: {CountParams(transformer):N0} params");
        Console.WriteLine($"Total: {CountParams(vqvae) + CountParams(transformer):N0}");
        Console.WriteLine($"EMA: {cfg.UseEma}, GradientClip: {cfg.GradientClipNorm}, Checkpoints: {cfg.SaveCheckpoints}");
        Console.WriteLine();

        if (!Directory.Exists(cfg.DataPath) || Directory.GetFiles(cfg.DataPath, "*.wav", SearchOption.AllDirectories).Length == 0)
        {
            Console.WriteLine($"Place .wav files in '{cfg.DataPath}/' and run again.");
            SmokeTest();
            return;
        }

        var files = Directory.GetFiles(cfg.DataPath, "*.wav", SearchOption.AllDirectories);
        Console.WriteLine($"Found {files.Length} audio files");

        var dataset = new Data.AudioDataset(cfg.DataPath, cfg.SampleRate, cfg.AudioLengthSamples);
        var trainer = new Training.Trainer(vqvae, transformer, device, cfg);

        // Load checkpoints if requested
        if (args.Length > 0 && args[0] == "--resume")
        {
            var epoch = args.Length > 1 ? int.Parse(args[1]) : 10;
            trainer.LoadCheckpoint("vqvae", epoch);
            trainer.LoadCheckpoint("transformer", epoch);
        }

        // Create validation split
        Data.AudioDataset? valDataset = null;
        if (cfg.ValSplit > 0 && dataset.Count > 1)
        {
            var valCount = (int)(dataset.Count * cfg.ValSplit);
            var trainCount = (int)(dataset.Count - valCount);
            if (trainCount > 0 && valCount > 0)
            {
                Console.WriteLine($"Split: {trainCount} train, {valCount} val");
                valDataset = new Data.AudioDataset(cfg.DataPath, cfg.SampleRate, cfg.AudioLengthSamples);
                // valDataset re-reads the same files; we handle split via skip in trainer
            }
        }

        Console.WriteLine("\nTraining VQ-VAE...");
        trainer.TrainVQVAE(dataset, cfg.NumEpochs, cfg.BatchSize, valDataset);

        Console.WriteLine("\nTraining Transformer...");
        trainer.TrainTransformer(dataset, cfg.NumEpochs, cfg.BatchSize, valDataset);

        if (dataset.Count > 0)
        {
            var sample = dataset.GetAudio(0).to(device, true);
            var (_, codes, _) = vqvae.Encode(sample, cfg.UseEma);
            var audio = trainer.Generate(codes[.., ..10], 100, 0.8);
            Utils.AudioIO.WriteWav("generated.wav", audio, cfg.SampleRate);
            Console.WriteLine("Saved generated.wav");
        }
    }

    static long CountParams(nn.Module<Tensor, Tensor> m)
    {
        long total = 0;
        foreach (var p in m.parameters())
        {
            long sz = 1;
            foreach (var s in p.shape) sz *= s;
            total += sz;
        }
        return total;
    }

    static void SmokeTest()
    {
        using (no_grad())
        {
            var x = ones([3], dtype: float32);
            Console.WriteLine($"Tensor test OK: [{string.Join(", ", (x * 2).data<float>().ToArray())}]");
        }
    }
}
