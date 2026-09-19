import torch

from model import IMAGE_SIZE, MODELS_DIR, build_model

ONNX_FP32_PATH = MODELS_DIR / "model_fp32.onnx"


def main():
    model = build_model()
    dummy_input = torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE)

    torch.onnx.export(
        model,
        dummy_input,
        ONNX_FP32_PATH,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"Exported ONNX model to {ONNX_FP32_PATH} ({ONNX_FP32_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
