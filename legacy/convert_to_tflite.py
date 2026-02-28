"""Convert ONNX float model -> INT8 TFLite for Raspberry Pi Pico (TFLM).

Usage (use Python 3.9 env with tensorflow + onnx2tf):
  python convert_to_tflite.py --onnx-path outputs_deploy/w025/model_float_opset13_cubeai.onnx

Pipeline:
  1. ONNX -> TF SavedModel + float32 TFLite  (via onnx2tf)
  2. float32 TFLite -> INT8 TFLite           (via TFLiteConverter, representative dataset)
  3. INT8 TFLite -> C header                  (for embedding in Pico firmware)
"""
from __future__ import annotations

import argparse
import glob
import subprocess
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ONNX -> INT8 TFLite converter")
    p.add_argument("--onnx-path", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("./outputs_deploy/tflite"))
    p.add_argument("--calib-samples", type=int, default=200,
                    help="Number of random calibration samples for INT8 quantization")
    return p.parse_args()


def step1_onnx_to_tflite_float(onnx_path: Path, saved_model_dir: Path) -> Path:
    """ONNX -> TF SavedModel + float32 TFLite via onnx2tf."""
    print(f"[1/3] ONNX -> TF SavedModel + float TFLite at {saved_model_dir}")
    saved_model_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "onnx2tf",
        "-i", str(onnx_path),
        "-o", str(saved_model_dir),
        "--non_verbose",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"STDOUT: {result.stdout[-1000:]}")
        print(f"STDERR: {result.stderr[-1000:]}")
        raise RuntimeError("onnx2tf failed")

    # onnx2tf auto-generates *_float32.tflite in the output dir
    float_tflites = list(saved_model_dir.glob("*_float32.tflite"))
    if not float_tflites:
        raise FileNotFoundError("onnx2tf did not produce a float32 .tflite file")
    float_tflite = float_tflites[0]
    print(f"  Done. Float TFLite: {float_tflite} ({float_tflite.stat().st_size/1024:.1f} KB)")
    return float_tflite


def step2_quantize_tflite_int8(float_tflite_path: Path, int8_tflite_path: Path,
                                calib_samples: int) -> None:
    """Quantize float32 TFLite -> full INT8 TFLite with representative dataset."""
    import tensorflow as tf

    print(f"[2/3] float32 TFLite -> INT8 TFLite at {int8_tflite_path}")

    # Load the float model to find actual input shape
    interpreter = tf.lite.Interpreter(model_path=str(float_tflite_path))
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']  # e.g. [1, 62, 13, 1]
    print(f"  Input shape: {input_shape}")

    def representative_dataset():
        for _ in range(calib_samples):
            data = np.random.randn(*input_shape).astype(np.float32)
            yield [data]

    # Quantize from the float TFLite directly
    converter = tf.lite.TFLiteConverter.from_saved_model(
        str(float_tflite_path.parent))

    # If from_saved_model fails (no signature), use interpreter approach
    try:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
    except (ValueError, Exception):
        print("  SavedModel conversion failed, using TFLite interpreter approach...")
        # Alternative: load the float .tflite and re-quantize via concrete functions
        # Build a simple tf.function wrapper around the tflite model
        model_content = float_tflite_path.read_bytes()

        # Use the simplest approach: convert from the tflite buffer
        # TF 2.15 doesn't support from_tflite directly, so use saved model with signatures
        # Fallback: use the saved model directory but add a serving signature
        import tensorflow as tf
        loaded = tf.saved_model.load(str(float_tflite_path.parent))

        # Try getting the concrete function
        concrete_func = None
        if hasattr(loaded, 'signatures') and 'serving_default' in loaded.signatures:
            concrete_func = loaded.signatures['serving_default']
        elif hasattr(loaded, '__call__'):
            # Trace the model
            @tf.function(input_signature=[tf.TensorSpec(shape=input_shape, dtype=tf.float32)])
            def serve(x):
                return loaded(x)
            concrete_func = serve.get_concrete_function()

        if concrete_func is not None:
            converter2 = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
            converter2.optimizations = [tf.lite.Optimize.DEFAULT]
            converter2.representative_dataset = representative_dataset
            converter2.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter2.inference_input_type = tf.int8
            converter2.inference_output_type = tf.int8
            tflite_model = converter2.convert()
        else:
            raise RuntimeError("Cannot convert model - no callable signature found")

    int8_tflite_path.write_bytes(tflite_model)
    size_kb = len(tflite_model) / 1024
    print(f"  Saved INT8 TFLite: {int8_tflite_path} ({size_kb:.1f} KB)")


def step3_tflite_to_c_header(tflite_path: Path, header_path: Path) -> None:
    """Convert .tflite binary to a C header file for embedding in firmware."""
    print(f"[3/3] TFLite -> C header at {header_path}")
    data = tflite_path.read_bytes()
    var_name = "kws_model_int8"

    lines = [
        f"// Auto-generated from {tflite_path.name}",
        f"// Model size: {len(data)} bytes ({len(data)/1024:.1f} KB)",
        f"#ifndef KWS_MODEL_H",
        f"#define KWS_MODEL_H",
        f"",
        f"#include <stdint.h>",
        f"",
        f"alignas(16) const uint8_t {var_name}[] = {{",
    ]

    # Write hex bytes, 16 per line
    for i in range(0, len(data), 16):
        chunk = data[i:i+16]
        hex_str = ", ".join(f"0x{b:02x}" for b in chunk)
        lines.append(f"  {hex_str},")

    lines.append("};")
    lines.append(f"const unsigned int {var_name}_len = {len(data)};")
    lines.append("")
    lines.append("#endif // KWS_MODEL_H")
    lines.append("")

    header_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved C header: {header_path}")
    print(f"  Array name: {var_name}, size: {len(data)} bytes")


def main() -> None:
    args = parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_model_dir = out_dir / "tf_saved_model"
    int8_tflite_path = out_dir / "model_int8.tflite"
    header_path = out_dir / "kws_model.h"

    float_tflite = step1_onnx_to_tflite_float(args.onnx_path, saved_model_dir)
    step2_quantize_tflite_int8(float_tflite, int8_tflite_path, args.calib_samples)
    step3_tflite_to_c_header(int8_tflite_path, header_path)

    print(f"\nAll done! Artifacts in {out_dir}/")
    print(f"  model_int8.tflite  - for testing on PC")
    print(f"  kws_model.h        - embed in Pico firmware")


if __name__ == "__main__":
    main()
