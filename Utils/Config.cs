namespace Eve.Utils;

public class Config
{
    public int SampleRate { get; set; } = 24000;
    public int AudioLengthSeconds { get; set; } = 5;
    public int AudioLengthSamples => SampleRate * AudioLengthSeconds;

    public int CodebookSize { get; set; } = 1024;
    public int LatentDim { get; set; } = 64;
    public int NumQuantizers { get; set; } = 8;

    public int EmbedDim { get; set; } = 512;
    public int NumHeads { get; set; } = 8;
    public int NumLayers { get; set; } = 6;
    public int FeedForwardDim { get; set; } = 2048;
    public int MaxSeqLen { get; set; } = 2048;

    public int BatchSize { get; set; } = 4;
    public float LearningRate { get; set; } = 1e-4f;
    public int NumEpochs { get; set; } = 10;
    public string DataPath { get; set; } = "./data";

    public string Device { get; set; } = "cpu";

    public bool UseEma { get; set; } = true;
    public float EmaDecay { get; set; } = 0.99f;
    public float EmaEpsilon { get; set; } = 1e-5f;
    public float CommitmentCost { get; set; } = 0.25f;

    public float GradientClipNorm { get; set; } = 1.0f;
    public int WarmupSteps { get; set; } = 1000;

    public bool SaveCheckpoints { get; set; } = true;
    public int SaveEvery { get; set; } = 5;
    public string CheckpointPath { get; set; } = "./checkpoints";

    public float ValSplit { get; set; } = 0.1f;
}
