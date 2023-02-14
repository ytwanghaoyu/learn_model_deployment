import numpy as np
import onnxruntime
import torch
import torch.functional as F
import torch.nn as nn
import torch.onnx as onnx
import torchvision
from torch.onnx.symbolic_registry import register_op


def support_ATen():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.asinh(x)

    def asinh_symbolic(g, input, *, out=None):
        return g.op("Asinh", input)

    register_op("asinh", asinh_symbolic, "", 9)

    model = Model()
    input = torch.rand(1, 3, 10, 10)
    torch_output = model(input).detach().numpy()
    torch.onnx.export(model, input, "asinh.onnx")

    sess = onnxruntime.InferenceSession("asinh.onnx")
    ort_output = sess.run(None, {"onnx::Asinh_0": input.numpy()})[0]

    assert np.allclose(torch_output, ort_output)


def support_TorchScript():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 18, 3)
            self.conv2 = torchvision.ops.DeformConv2d(3, 3, 3)

        def forward(self, x):
            return self.conv2(x, self.conv1(x))

    @parse_args("v", "v", "v", "v", "v", "i", "i", "i", "i", "i", "i", "i", "i", "none")
    def symbolic(
        g,
        input,
        weight,
        offset,
        mask,
        bias,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dil_h,
        dil_w,
        n_weight_grps,
        n_offset_grps,
        use_mask,
    ):
        return g.op("custom::deform_conv2d", input, offset)


if __name__ == "__main__":
    # support_ATen()
    support_TorchScript()
