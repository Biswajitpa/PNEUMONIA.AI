import os
import tensorflow as tf
import tf2onnx

# ── PATH CONFIGURATION ──────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
H5_MODEL_PATH = os.path.join(BASE_DIR, "storage", "models", "xray_model_best.h5")
ONNX_MODEL_PATH = os.path.join(BASE_DIR, "storage", "models", "xray_model_best.onnx")

def convert_h5_to_onnx():
    # 1. Verify source weights exist
    if not os.path.exists(H5_MODEL_PATH):
        print(f"❌ Error: Source weights file not found at: {H5_MODEL_PATH}")
        print("Please ensure your 'xray_model_best.h5' file is placed inside 'storage/models/' first.")
        return

    # Ensure target directory path exists
    os.makedirs(os.path.dirname(ONNX_MODEL_PATH), exist_ok=True)

    print("🧠 Loading original H5 Keras model weights (excluding training state)...")
    try:
        # load_model with compile=False avoids optimization/loss tracking layers overhead
        model = tf.keras.models.load_model(H5_MODEL_PATH, compile=False)
    except Exception as e:
        print(f"❌ Failed to parse Keras model: {e}")
        return

    print("⚡ Defining image tensor input signatures [None, 224, 224, 3]...")
    # 'None' allows for dynamic evaluation of varying batch run sizes during runtime inference
    input_signature = [tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input")]

    print("🚀 Initializing framework signature conversion to ONNX format...")
    try:
        model_proto, _ = tf2onnx.convert.from_keras(
            model, 
            input_signature=input_signature, 
            output_path=ONNX_MODEL_PATH
        )
        print(f"✅ Success! Cleanly compiled model optimization binary to:\n   👉 {ONNX_MODEL_PATH}")
        print("\nNow you can commit this file or run it locally to generate your ONNX asset.")
    except Exception as e:
        print(f"❌ ONNX translation compilation broke down: {e}")

if __name__ == "__main__":
    convert_h5_to_onnx()
