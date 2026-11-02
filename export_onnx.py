import argparse

import numpy as np
import onnxruntime as ort
import torch
from torch import nn
from torchvision import models


def export(model_path, out_path, labels_path):
    ckpt = torch.load(model_path, map_location="cpu")
    classes = ckpt["classes"]

    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ckpt["model"])
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )

    with open(labels_path, "w") as f:
        f.write("\n".join(classes))

    with torch.no_grad():
        expected = model(dummy).numpy()
    session = ort.InferenceSession(out_path)
    actual = session.run(None, {"input": dummy.numpy()})[0]
    print("max diff:", np.abs(expected - actual).max())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--out", default="model.onnx")
    parser.add_argument("--labels", default="labels.txt")
    args = parser.parse_args()
    export(args.model_path, args.out, args.labels)
