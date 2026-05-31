using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace Eve.Training;

public class Trainer
{
    private readonly Models.VQVAE _vqvae;
    private readonly Models.AudioTransformer _transformer;
    private readonly optim.Optimizer _vqvaeOpt;
    private readonly optim.Optimizer _transformerOpt;
    private readonly Device _device;
    private readonly Utils.Config _config;
    private int _step;

    public Trainer(Models.VQVAE vqvae, Models.AudioTransformer transformer, Device device, Utils.Config config)
    {
        _vqvae = vqvae;
        _transformer = transformer;
        _device = device;
        _config = config;

        _vqvae = (_vqvae.to(_device) as Models.VQVAE)!;
        _transformer = (_transformer.to(_device) as Models.AudioTransformer)!;

        _vqvaeOpt = optim.Adam(_vqvae.parameters(), config.LearningRate);
        _transformerOpt = optim.Adam(_transformer.parameters(), config.LearningRate);
    }

    public (float trainLoss, float valLoss) TrainVQVAE(Data.AudioDataset dataset, int epochs, int batchSize, Data.AudioDataset? valDataset = null)
    {
        _vqvae.train();
        var valCount = valDataset?.Count ?? 0;
        var hasVal = valCount > 0;

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            float epochLoss = 0;
            int batchCount = 0;

            for (int i = 0; i < dataset.Count; i += batchSize)
            {
                var end = (int)Math.Min(i + batchSize, dataset.Count);
                var tensors = new List<Tensor>();
                for (int j = i; j < end; j++)
                    tensors.Add(dataset.GetAudio(j));

                var batch = stack(tensors).to(_device, true);
                _vqvaeOpt.zero_grad();

                var z = _vqvae.GetEncoderOutput(batch);
                var (q, _, commitLoss) = _vqvae.Quantizer.Quantize(z, _config.UseEma);
                var recon = _vqvae.DecoderNet.forward(q);
                var loss = nn.functional.mse_loss(recon, batch) + commitLoss;

                loss.backward();

                if (_config.GradientClipNorm > 0)
                    nn.utils.clip_grad_norm_(_vqvae.parameters(), _config.GradientClipNorm);

                _vqvaeOpt.step();

                if (_config.UseEma)
                    _vqvae.Quantizer.UpdateEma(z.detach());

                epochLoss += loss.item<float>();
                batchCount++;
                _step++;
            }

            if (hasVal)
            {
                float valLoss = 0;
                int valCount2 = 0;
                _vqvae.eval();
                using (no_grad())
                for (int i = 0; i < valDataset!.Count; i += batchSize)
                {
                    var end = (int)Math.Min(i + batchSize, valDataset.Count);
                    var tensors = new List<Tensor>();
                    for (int j = i; j < end; j++)
                        tensors.Add(valDataset.GetAudio(j));

                    var batch = stack(tensors).to(_device, true);
                    var (q, _, commitLoss) = _vqvae.Encode(batch, _config.UseEma);
                    var recon = _vqvae.DecoderNet.forward(q);
                    var loss = nn.functional.mse_loss(recon, batch) + commitLoss;
                    valLoss += loss.item<float>();
                    valCount2++;
                }
                _vqvae.train();
                Console.WriteLine($"  Epoch {epoch + 1}/{epochs}  train: {epochLoss / batchCount:F6}  val: {valLoss / valCount2:F6}");
                if (_config.SaveCheckpoints && (epoch + 1) % _config.SaveEvery == 0)
                    SaveCheckpoint("vqvae", epoch + 1);
            }
            else
            {
                Console.WriteLine($"  Epoch {epoch + 1}/{epochs}  loss: {epochLoss / batchCount:F6}");
                if (_config.SaveCheckpoints && (epoch + 1) % _config.SaveEvery == 0)
                    SaveCheckpoint("vqvae", epoch + 1);
            }
        }
        return (0, 0); // caller can ignore
    }

    public (float trainLoss, float valLoss) TrainTransformer(Data.AudioDataset dataset, int epochs, int batchSize, Data.AudioDataset? valDataset = null)
    {
        _vqvae.eval();
        _transformer.train();
        var hasVal = valDataset?.Count > 0;

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            float epochLoss = 0;
            int batchCount = 0;

            for (int i = 0; i < dataset.Count; i += batchSize)
            {
                var end = (int)Math.Min(i + batchSize, dataset.Count);
                var allCodes = new List<Tensor>();
                for (int j = i; j < end; j++)
                {
                    var audio = dataset.GetAudio(j).to(_device, true);
                    var (_, codes, _) = _vqvae.Encode(audio, _config.UseEma);
                    allCodes.Add(codes);
                }

                var codeBatch = stack(allCodes);
                var input = codeBatch[.., ..^-1];
                var target = codeBatch[.., 1..];

                _transformerOpt.zero_grad();
                var logits = _transformer.forward(input);
                var loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[^1]),
                    target.reshape(-1));

                loss.backward();

                if (_config.GradientClipNorm > 0)
                    nn.utils.clip_grad_norm_(_transformer.parameters(), _config.GradientClipNorm);

                _transformerOpt.step();

                epochLoss += loss.item<float>();
                batchCount++;
                _step++;
            }

            if (hasVal)
            {
                float valLoss = 0;
                int valCount2 = 0;
                _transformer.eval();
                using (no_grad())
                for (int i = 0; i < valDataset!.Count; i += batchSize)
                {
                    var end = (int)Math.Min(i + batchSize, valDataset.Count);
                    var allCodes = new List<Tensor>();
                    for (int j = i; j < end; j++)
                    {
                        var audio = valDataset.GetAudio(j).to(_device, true);
                        var (_, codes, _) = _vqvae.Encode(audio, _config.UseEma);
                        allCodes.Add(codes);
                    }

                    var codeBatch = stack(allCodes);
                    var input = codeBatch[.., ..^-1];
                    var target = codeBatch[.., 1..];

                    var logits = _transformer.forward(input);
                    var loss = nn.functional.cross_entropy(
                        logits.reshape(-1, logits.shape[^1]),
                        target.reshape(-1));

                    valLoss += loss.item<float>();
                    valCount2++;
                }
                _transformer.train();
                Console.WriteLine($"  Epoch {epoch + 1}/{epochs}  train: {epochLoss / batchCount:F6}  val: {valLoss / valCount2:F6}");
                if (_config.SaveCheckpoints && (epoch + 1) % _config.SaveEvery == 0)
                    SaveCheckpoint("transformer", epoch + 1);
            }
            else
            {
                Console.WriteLine($"  Epoch {epoch + 1}/{epochs}  loss: {epochLoss / batchCount:F6}");
                if (_config.SaveCheckpoints && (epoch + 1) % _config.SaveEvery == 0)
                    SaveCheckpoint("transformer", epoch + 1);
            }
        }
        return (0, 0);
    }

    public Tensor Generate(Tensor prompt, long steps, double temperature = 1.0)
    {
        _vqvae.eval();
        _transformer.eval();
        using (no_grad())
        {
            var codes = _transformer.Generate(prompt, steps, temperature);
            return _vqvae.DecodeFromCodes(codes);
        }
    }

    public void SaveCheckpoint(string prefix, int epoch)
    {
        var baseDir = _config.CheckpointPath;
        if (!Directory.Exists(baseDir))
            Directory.CreateDirectory(baseDir);

        if (prefix is "vqvae" or "both")
        {
            var dir = Path.Combine(baseDir, $"vqvae_ep{epoch}");
            Directory.CreateDirectory(dir);
            foreach (var (k, v) in _vqvae.state_dict())
                torch.save(v, Path.Combine(dir, $"{k.Replace('/', '_')}.pt"));
            Console.WriteLine($"Saved VQ-VAE checkpoint: {dir}");
        }
        if (prefix is "transformer" or "both")
        {
            var dir = Path.Combine(baseDir, $"transformer_ep{epoch}");
            Directory.CreateDirectory(dir);
            foreach (var (k, v) in _transformer.state_dict())
                torch.save(v, Path.Combine(dir, $"{k.Replace('/', '_')}.pt"));
            Console.WriteLine($"Saved transformer checkpoint: {dir}");
        }
    }

    public void LoadCheckpoint(string prefix, int epoch)
    {
        if (prefix is "vqvae" or "both")
        {
            var dir = Path.Combine(_config.CheckpointPath, $"vqvae_ep{epoch}");
            if (Directory.Exists(dir))
            {
                var sd = new Dictionary<string, Tensor>();
                foreach (var f in Directory.GetFiles(dir, "*.pt"))
                    sd[Path.GetFileNameWithoutExtension(f).Replace('_', '/')] = torch.load(f);
                _vqvae.load_state_dict(sd);
                Console.WriteLine($"Loaded VQ-VAE checkpoint: {dir}");
            }
        }
        if (prefix is "transformer" or "both")
        {
            var dir = Path.Combine(_config.CheckpointPath, $"transformer_ep{epoch}");
            if (Directory.Exists(dir))
            {
                var sd = new Dictionary<string, Tensor>();
                foreach (var f in Directory.GetFiles(dir, "*.pt"))
                    sd[Path.GetFileNameWithoutExtension(f).Replace('_', '/')] = torch.load(f);
                _transformer.load_state_dict(sd);
                Console.WriteLine($"Loaded transformer checkpoint: {dir}");
            }
        }
    }
}
