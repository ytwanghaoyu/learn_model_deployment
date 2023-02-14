import os

import cv2
import numpy as np
import torch
import torch.onnx
from torch import nn
from torch.nn.functional import interpolate


class NewInterpolate(torch.autograd.Function):
    @staticmethod
    def symbolic(g, input, scales):
        return g.op(
            "Resize",
            input,
            g.op("Constant", value_t=torch.tensor([], dtype=torch.float32)),
            scales,
            coordinate_transformation_mode_s="pytorch_half_pixel",
            cubic_coeff_a_f=-0.75,
            mode_s="cubic",
            nearest_mode_s="floor",
        )

    @staticmethod
    def forward(ctx, input, scales):
        scales = scales.tolist()[-2:]
        return interpolate(
            input, scale_factor=scales, mode="bicubic", align_corners=False
        )


class StrangeSuperResolutionNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

        self.relu = nn.ReLU()

    def forward(self, x, upscale_factor):
        x = NewInterpolate.apply(x, upscale_factor)
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return out


class SuperResolutionNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

        self.relu = nn.ReLU()

    def forward(self, x, upscale_factor):
        # x = interpolate(
        #     x, scale_factor=upscale_factor, mode="bicubic", align_corners=False
        # )
        x = interpolate(
            x, scale_factor=upscale_factor.item(), mode="bicubic", align_corners=False
        )
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return out


def init_torch_model():
    torch_model = StrangeSuperResolutionNet()
    # torch_model = SuperResolutionNet()

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


def deploy_with_onnx_runtime(input_img, onnx_path="srcnn.onnx"):
    import onnxruntime

    ort_session = onnxruntime.InferenceSession(onnx_path)
    ort_inputs = {"input": input_img}
    ort_output = ort_session.run(["output"], ort_inputs)[0]
    return ort_output


def deploy_with_onnx_runtime_new(input_img, onnx_path="srcnn3.onnx"):
    import onnxruntime

    input_factor = np.array([1, 1, 4, 4], dtype=np.float32)
    ort_session = onnxruntime.InferenceSession(onnx_path)
    ort_inputs = {"input": input_img, "factor": input_factor}
    ort_output = ort_session.run(None, ort_inputs)[0]
    return ort_output


if __name__ == "__main__":
    model = init_torch_model()
    input_img = process_img_input(img_path="face.png")

    # Inference
    # torch_output = model(torch.from_numpy(input_img), 3).detach().numpy()

    # Note that the second input is torch.tensor(3)
    # torch_output = model(torch.from_numpy(input_img), torch.tensor(3)).detach().numpy()

    # Note that the factor is changed to 1,1,3,3
    factor = torch.tensor([1, 1, 3, 3], dtype=torch.float)
    torch_output = model(torch.from_numpy(input_img), factor).detach().numpy()

    process_img_output(torch_output, output_path="face_torch_3.png")

    # Gen dummy input
    x = torch.randn(1, 3, 256, 256)

    # Export to ONNX
    # with torch.no_grad():
    #     torch.onnx.export(
    #         model,
    #         (x, 3),
    #         "srcnn.onnx",
    #         opset_version=11,
    #         input_names=["input", "factor"],
    #         output_names=["output"],
    #     )

    # with torch.no_grad():
    #     torch.onnx.export(
    #         model,
    #         (x, torch.tensor(3)),
    #         "srcnn2.onnx",
    #         opset_version=11,
    #         input_names=["input", "factor"],
    #         output_names=["output"],
    #     )

    with torch.no_grad():
        torch.onnx.export(
            model,
            (x, factor),
            "srcnn3.onnx",
            opset_version=11,
            input_names=["input", "factor"],
            output_names=["output"],
        )

    # # Check ONNX file
    # check_onnx(onnx_path="srcnn.onnx")

    # Run with ONNX runtime
    ort_output = deploy_with_onnx_runtime_new(input_img, onnx_path="srcnn3.onnx")
    process_img_output(ort_output, output_path="face_ort_3.png")
