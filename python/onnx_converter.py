#!/usr/bin/env python3
"""Export a PyTorch model to ONNX.

The model is defined in `build_model()` below - edit that function to export your own
network. The default is torchvision's FCN-ResNet50 (semantic segmentation).

Example:
    python python/onnx_converter.py --output models/fcn.onnx --input-shape 1 3 512 512 --dynamic
"""
import argparse
from pathlib import Path

import onnx
import torch
from torchvision.models.segmentation import FCN_ResNet50_Weights, fcn_resnet50


class _TensorOutput(torch.nn.Module):
    """torchvision segmentation models return a dict; ONNX export needs a plain tensor."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["out"]


def build_model(weights_path=None):
    """Return the model to export, in eval mode.

    With `weights_path` the checkpoint is loaded on top of an untrained network;
    otherwise the pretrained torchvision weights are used.
    """
    if weights_path:
        model = fcn_resnet50(weights=None, weights_backbone=None)
        checkpoint = torch.load(weights_path, map_location="cpu")
        # Checkpoints are often wrapped, e.g. {"state_dict": ...}; unwrap the first known key.
        for key in ("model_param", "state_dict", "model_state"):
            if isinstance(checkpoint, dict) and key in checkpoint:
                print(f"Using checkpoint key '{key}'")
                checkpoint = checkpoint[key]
                break
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded weights from {weights_path}")
    else:
        model = fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT)
    return _TensorOutput(model).eval()


def export_onnx(model, input_shape, output_path, dynamic=False):
    dummy_input = torch.randn(*input_shape)

    dynamic_axes = None
    if dynamic:
        # Batch, height and width stay dynamic; the TensorRT min/opt/max range limits them later.
        dynamic_axes = {
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print(f"Saved ONNX model to {output_path}")
    print("Inputs: ", [i.name for i in onnx_model.graph.input])
    print("Outputs:", [o.name for o in onnx_model.graph.output])


def main():
    parser = argparse.ArgumentParser(description="Export a PyTorch model to ONNX.")
    parser.add_argument("--output", required=True, help="where to write the .onnx file")
    parser.add_argument("--input-shape", type=int, nargs=4, metavar=("N", "C", "H", "W"),
                        default=[1, 3, 512, 512], help="example input shape (default: 1 3 512 512)")
    parser.add_argument("--dynamic", action="store_true", help="make batch/height/width dynamic")
    parser.add_argument("--weights", help="optional PyTorch checkpoint to load")
    args = parser.parse_args()

    export_onnx(build_model(args.weights), args.input_shape, args.output, args.dynamic)


if __name__ == "__main__":
    main()
