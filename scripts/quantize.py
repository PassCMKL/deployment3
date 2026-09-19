from collections import Counter

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic

from model import MODELS_DIR

ONNX_FP32_PATH = MODELS_DIR / "model_fp32.onnx"
ONNX_INT8_PATH = MODELS_DIR / "model_int8.onnx"


def op_type_counts(path):
    model = onnx.load(str(path))
    return Counter(node.op_type for node in model.graph.node)


def main():
    quantize_dynamic(
        model_input=str(ONNX_FP32_PATH),
        model_output=str(ONNX_INT8_PATH),
        weight_type=QuantType.QUInt8,
    )

    fp32_size = ONNX_FP32_PATH.stat().st_size / 1e6
    int8_size = ONNX_INT8_PATH.stat().st_size / 1e6
    print(f"FP32 ONNX: {fp32_size:.1f} MB")
    print(f"INT8 ONNX: {int8_size:.1f} MB ({fp32_size / int8_size:.1f}x smaller)")

    print("\nOp types before quantization:")
    print(op_type_counts(ONNX_FP32_PATH))
    print("\nOp types after quantization:")
    print(op_type_counts(ONNX_INT8_PATH))


if __name__ == "__main__":
    main()
