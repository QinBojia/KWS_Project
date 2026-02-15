"""
Robust ONNX to TFLite Int8 Converter
------------------------------------
1. Converts ONNX -> TF SavedModel (using onnx2tf)
2. Converts SavedModel -> TFLite Int8 (using TF Lite Converter)
"""

import os
import shutil
import subprocess
import sys
import numpy as np
import tensorflow as tf
import onnx
from config import AudioConfig
from data import SpeechCommandsMFCC12

# ---------------- CONFIGURATION ---------------- #
ONNX_PATH = "outputs_deploy/w025_cubeai/model_float.onnx"
OUTPUT_DIR = "outputs_deploy/w025_cubeai/tflite_int8_clean"
CALIB_SAMPLES = 200  # Number of samples for quantization calibration
# ----------------------------------------------- #

def check_model_input_name(onnx_path):
    print(f"[*] Checking input name for {onnx_path}...")
    model = onnx.load(onnx_path)
    input_names = [node.name for node in model.graph.input]
    print(f"    Found inputs: {input_names}")
    return input_names[0]

def run_onnx2tf(onnx_path, saved_model_dir):
    """
    Run onnx2tf as a subprocess.
    Using -kt (Keep Tensor) usually helps with 1D/Audio input shapes to prevent incorrect transposes.
    """
    if os.path.exists(saved_model_dir):
        shutil.rmtree(saved_model_dir)
    
    print(f"[*] converting ONNX to TF SavedModel...")
    cmd = [
        sys.executable, "-m", "onnx2tf",
        "-i", onnx_path,
        "-o", saved_model_dir,
        "-kt",  # Keep Tensor input/output layout (crucial for audio/1D layers)
        "--non_verbose"
    ]
    
    print(f"    Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print("[*] Conversion to SavedModel successful.")

def representative_dataset_gen(audio_cfg, n_samples=100):
    """
    Generator function for quantization calibration.
    Reads real data from the validation set.
    """
    # Disable caching to ensure fresh read if needed, or use existing cache
    ds = SpeechCommandsMFCC12(
        subset="validation",
        audio_cfg=audio_cfg,
        silence_ratio=0.0,
        cache_dir="./cache_mfcc",
        use_cache=True,
    )
    
    print(f"[*] Generating {n_samples} calibration samples from {len(ds)} validation files...")
    
    count = 0
    for i in range(len(ds)):
        if count >= n_samples:
            break
        
        # Get data: (frames, n_mfcc)
        x, _ = ds[i]
        
        # Add batch dim -> (1, 1, frames, n_mfcc) to match ONNX NCHW layout
        # Note: onnx2tf might have transposed the model to NHWC internally, 
        # but usually expects inputs in the original format or standard TF format.
        # Let's inspect the SavedModel signature later if this fails, but typically
        # we provide the shape the model expects.
        
        # The model input is (1, 1, 62, 13). 
        x = x.unsqueeze(0).unsqueeze(0) 
        
        # Yield as list of tensors
        yield [x.numpy()]
        count += 1

def convert_to_tflite_int8(saved_model_dir, tflite_path, audio_cfg):
    print(f"[*] Converting SavedModel to TFLite (Int8)...")
    
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    
    # Enable optimizations (quantization)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Set representative dataset for full integer quantization
    converter.representative_dataset = lambda: representative_dataset_gen(audio_cfg, CALIB_SAMPLES)
    
    # Ensure full integer quantization compatibility
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_model = converter.convert()
    
    # Save
    os.makedirs(os.path.dirname(tflite_path), exist_ok=True)
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
        
    print(f"[*] Success! TFLite model saved to: {tflite_path}")

def main():
    if not os.path.exists(ONNX_PATH):
        print(f"Error: ONNX file not found at {ONNX_PATH}")
        sys.exit(1)

    # Setup directories
    saved_model_dir = os.path.join(OUTPUT_DIR, "tf_saved_model")
    tflite_path = os.path.join(OUTPUT_DIR, "model_int8.tflite")
    
    # 0. Check Input Name (sanity check)
    check_model_input_name(ONNX_PATH)

    # 1. Convert ONNX -> TF SavedModel
    try:
        run_onnx2tf(ONNX_PATH, saved_model_dir)
    except subprocess.CalledProcessError as e:
        print("Error during onnx2tf conversion. Make sure onnx2tf is installed.")
        sys.exit(1)

    # 2. Config for Data Loading
    audio_cfg = AudioConfig()

    # 3. Convert TF SavedModel -> TFLite
    convert_to_tflite_int8(saved_model_dir, tflite_path, audio_cfg)

if __name__ == "__main__":
    main()

