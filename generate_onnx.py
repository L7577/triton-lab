"""Generate a tiny ONNX model for Triton benchmarking."""
import onnx
from onnx import helper, TensorProto

INPUT_SHAPE = ["batch", 256, 256]  # Dynamic batch dim for Triton dynamic batching

def make_identity_model():
    """Create an identity model: input → Identity → output."""
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, INPUT_SHAPE
    )
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, INPUT_SHAPE
    )

    node = helper.make_node(
        "Identity", inputs=["input"], outputs=["output"], name="identity"
    )

    graph = helper.make_graph(
        [node], "identity_graph", [input_tensor], [output_tensor]
    )

    model = helper.make_model(graph, producer_name="triton-lab", opset_imports=[
        helper.make_operatorsetid("", 8)
    ])
    model.ir_version = 6  # Compatible with Triton 22.12's ONNX Runtime

    onnx.checker.check_model(model)
    return model


if __name__ == "__main__":
    import sys, os
    out = sys.argv[1] if len(sys.argv) > 1 else "identity.onnx"
    model = make_identity_model()
    onnx.save(model, out)
    print(f"Generated {out} ({os.path.getsize(out)} bytes)")
