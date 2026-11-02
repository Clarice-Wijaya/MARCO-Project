using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

float[] mean = { 0.485f, 0.456f, 0.406f };
float[] std = { 0.229f, 0.224f, 0.225f };

var labels = File.ReadAllLines("labels.txt");
using var session = new InferenceSession("model.onnx");

foreach (var path in args)
{
    var input = LoadImage(path);
    var inputs = new[] { NamedOnnxValue.CreateFromTensor("input", input) };
    using var results = session.Run(inputs);

    var probs = Softmax(results[0].AsEnumerable<float>().ToArray());
    int best = Array.IndexOf(probs, probs.Max());
    Console.WriteLine($"{Path.GetFileName(path)}: {labels[best]} ({probs[best]:P1})");
}

DenseTensor<float> LoadImage(string path)
{
    using var image = Image.Load<Rgb24>(path);
    image.Mutate(x => x.Resize(new ResizeOptions
    {
        Size = new Size(224, 224),
        Mode = ResizeMode.Stretch,
        Sampler = KnownResamplers.Triangle,
    }));

    var tensor = new DenseTensor<float>(new[] { 1, 3, 224, 224 });
    for (int y = 0; y < 224; y++)
    {
        for (int x = 0; x < 224; x++)
        {
            var p = image[x, y];
            tensor[0, 0, y, x] = (p.R / 255f - mean[0]) / std[0];
            tensor[0, 1, y, x] = (p.G / 255f - mean[1]) / std[1];
            tensor[0, 2, y, x] = (p.B / 255f - mean[2]) / std[2];
        }
    }
    return tensor;
}

float[] Softmax(float[] logits)
{
    float max = logits.Max();
    var exp = logits.Select(v => MathF.Exp(v - max)).ToArray();
    float sum = exp.Sum();
    return exp.Select(v => v / sum).ToArray();
}