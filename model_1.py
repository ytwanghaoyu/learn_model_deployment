import os

import cv2
import numpy as np
import requests
import torch
import torch.onnx
from torch import nn


class SuperResolutionNet(nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.img_upsampler = nn.Upsample(
            scale_factor=self.upscale_factor, mode="bicubic", align_corners=False
        )

        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.img_upsampler(x)
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return out


def init_torch_model():
    torch_model = SuperResolutionNet(upscale_factor=3)

    state_dict = torch.load("srcnn.pth")["state_dict"]

    # Adapt the checkpoint
    for old_key in list(state_dict.keys()):
        new_key = ".".join(old_key.split(".")[1:])
        state_dict[new_key] = state_dict.pop(old_key)

    torch_model.load_state_dict(state_dict)
    torch_model.eval()
    return torch_model


def process_img_input(img_path="face_input.png"):
    input_img = cv2.imread(img_path).astype(np.float32)

    # HWC to NCHW
    input_img = np.transpose(input_img, [2, 0, 1])
    input_img = np.expand_dims(input_img, 0)
    return input_img


def process_img_output(input_img, output_path="face_output.png"):
    # NCHW to HWC
    output_img = np.squeeze(input_img, 0)
    output_img = np.clip(output_img, 0, 255)
    output_img = np.transpose(output_img, [1, 2, 0]).astype(np.uint8)

    # Show image
    if output_path is not None:
        cv2.imwrite(output_path, output_img)
    return output_img


def check_onnx(onnx_path="srcnn.onnx"):
    import onnx

    onnx_model = onnx.load(onnx_path)
    try:
        onnx.checker.check_model(onnx_model)
    except Exception:
        print("Model incorrect")
    else:
        print("Model correct")


def deploy_with_onnx_runtime(input_img):
    import onnxruntime

    ort_session = onnxruntime.InferenceSession("srcnn.onnx")
    ort_inputs = {"input": input_img}
    ort_output = ort_session.run(["output"], ort_inputs)[0]
    return ort_output


if __name__ == "__main__":
    model = init_torch_model()
    input_img = process_img_input(img_path="face.png")

    # Inference
    torch_output = model(torch.from_numpy(input_img)).detach().numpy()

    process_img_output(torch_output, output_path="face_torch.png")

    # Gen dummy input
    x = torch.randn(1, 3, 256, 256)

    # Export to ONNX
    with torch.no_grad():
        torch.onnx.export(
            model,
            x,
            "srcnn.onnx",
            opset_version=11,
            input_names=["input"],
            output_names=["output"],
        )

    # Check ONNX file
    check_onnx(onnx_path="srcnn.onnx")

    # Run with ONNX runtime
    ort_output = deploy_with_onnx_runtime(input_img)
    process_img_output(ort_output, output_path="face_ort.png")
