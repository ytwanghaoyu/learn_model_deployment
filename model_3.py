import numpy as np
import onnxruntime
import torch


def try1():
    class Model(torch.nn.Module):
        def __init__(self, n):
            super().__init__()
            self.n = n
            self.conv = torch.nn.Conv2d(3, 3, 3)

        def forward(self, x):
            for i in range(self.n):
                x = self.conv(x)
            return x

    models = [Model(2), Model(3)]
    model_names = ["model_2", "model_3"]

    for model, model_name in zip(models, model_names):
        dummy_input = torch.rand(1, 3, 10, 10)
        dummy_output = model(dummy_input)
        model_trace = torch.jit.trace(model, dummy_input)
        model_script = torch.jit.script(model)

        # 跟踪法与直接 torch.onnx.export(model, ...)等价
        torch.onnx.export(
            model_trace,
            dummy_input,
            f"{model_name}_trace.onnx",
            example_outputs=dummy_output,
        )
        # 记录法必须先调用 torch.jit.sciprt
        torch.onnx.export(
            model_script,
            dummy_input,
            f"{model_name}_script.onnx",
            example_outputs=dummy_output,
        )


def try2():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, 3)

        def forward(self, x):
            x = self.conv(x)
            return x

    model = Model()
    dummy_input = torch.rand(1, 3, 10, 10)
    model_names = ["model_static.onnx", "model_dynamic_0.onnx", "model_dynamic_23.onnx"]

    dynamic_axes_0 = {"in": [0], "out": [0]}
    dynamic_axes_23 = {"in": [2, 3], "out": [2, 3]}

    torch.onnx.export(
        model, dummy_input, model_names[0], input_names=["in"], output_names=["out"]
    )
    torch.onnx.export(
        model,
        dummy_input,
        model_names[1],
        input_names=["in"],
        output_names=["out"],
        dynamic_axes=dynamic_axes_0,
    )
    torch.onnx.export(
        model,
        dummy_input,
        model_names[2],
        input_names=["in"],
        output_names=["out"],
        dynamic_axes=dynamic_axes_23,
    )

    origin_tensor = np.random.rand(1, 3, 10, 10).astype(np.float32)
    mult_batch_tensor = np.random.rand(2, 3, 10, 10).astype(np.float32)
    big_tensor = np.random.rand(1, 3, 20, 20).astype(np.float32)

    inputs = [origin_tensor, mult_batch_tensor, big_tensor]
    exceptions = dict()

    for model_name in model_names:
        for i, input in enumerate(inputs):
            try:
                ort_session = onnxruntime.InferenceSession(model_name)
                ort_inputs = {"in": input}
                ort_session.run(["out"], ort_inputs)
            except Exception as e:
                exceptions[(i, model_name)] = e
                print(f"Input[{i}] on model {model_name} error.")
            else:
                print(f"Input[{i}] on model {model_name} succeed.")


def try3():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, 3)

        def forward(self, x):
            x = self.conv(x)
            if torch.onnx.is_in_onnx_export():
                x = torch.clip(x, 0, 1)
            return x


def try4():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            x = x * x[0].item()
            return x, torch.Tensor([i for i in x])

    model = Model()
    dummy_input = torch.rand(10)
    torch.onnx.export(model, dummy_input, "a.onnx")


if __name__ == "__main__":
    # try1()
    try2()
