import os
import requests
import gradio as gr
import numpy as np
import cv2
from PIL import Image
from core.classifier import XRayClassifier
from core.gemini_client import GeminiMedicalClient
from core.report_generator import PDFReportGenerator

# ==========================================
# 1. PATH CONFIGURATION & FILE DOWNLOADER
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "storage", "models")

# Automatically build missing directories inside the Render container
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "xray_model_best.h5")

# Direct download route configuration using your unique File ID
FILE_ID = "1-XmVDb3ldcpbMd--OX_PkbzBkDgceI0q"

# CRITICAL FIX: Appended '&confirm=t' to cleanly bypass Google Drive's >100MB scanning restriction page
MODEL_URL = f"https://docs.google.com/uc?export=download&id={FILE_ID}&confirm=t"

if not os.path.exists(MODEL_PATH):
    print("🚀 Model weights missing from server storage.")
    print("📥 Initializing direct download from Google Drive...")
    try:
        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("✅ Model weights completely downloaded and saved locally!")
    except Exception as e:
        print(f"❌ Cloud download failed: {e}")

# ==========================================
# 2. CORE ENGINES INITIALIZATION
# ==========================================
print("🧠 Initializing ResNet50 Classifier model weights...")
classifier = XRayClassifier(MODEL_PATH)
gemini_engine = GeminiMedicalClient(api_key=os.getenv("GEMINI_API_KEY"))

# ==========================================
# 3. CLINICAL LOGIC PIPELINE
# ==========================================
def analyze_xray(image, name, age, gender):
    """Core logic to process X-ray and generate report."""
    if image is None:
        return None, "No image provided", "0%", "N/A"

    # Preprocessing
    img_array = np.array(image.convert('RGB'))
    img_resized = cv2.resize(img_array, (224, 224))
    img_tensor = np.expand_dims(img_resized, axis=0)

    # ML Prediction
    diagnosis, confidence = classifier.predict(img_tensor)
    
    # Grad-CAM logic
    if diagnosis.upper() == "PNEUMONIA":
        heatmap = classifier.generate_gradcam(img_tensor)
        gradcam_blend = cv2.addWeighted(img_resized, 0.6, heatmap, 0.4, 0)
        output_img = Image.fromarray(gradcam_blend)
        affected_part = "Consolidation detected in lung tissue"
    else:
        output_img = image
        affected_part = "Clear (No consolidation detected)"

    # AI Report Generation
    patient_info = {"name": name, "age": age, "gender": gender}
    ai_note = gemini_engine.generate_narrative(patient_info, diagnosis, confidence, affected_part)
    
    return output_img, diagnosis, f"{confidence:.2f}%", ai_note

# ==========================================
# 4. UI LAYOUT (GRADIO BLOCKS)
# ==========================================
with gr.Blocks(title="PNEUMONIA.AI") as demo:
    gr.Markdown("# 🧠 PNEUMONIA.AI - Clinical Diagnostic Support")
    gr.Markdown("Upload a chest X-ray to detect pneumonia and view the explainable heatmap.")
    
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="pil", label="Upload Chest X-Ray")
            name = gr.Textbox(label="Patient Name")
            age = gr.Number(label="Patient Age")
            gender = gr.Dropdown(["Male", "Female", "Other"], label="Gender")
            analyze_btn = gr.Button("Run Analysis", variant="primary")
        
        with gr.Column():
            img_output = gr.Image(label="Explainable Heatmap")
            diagnosis_out = gr.Textbox(label="Model Diagnosis")
            conf_out = gr.Textbox(label="Confidence Score")
            report_out = gr.Textbox(label="AI Clinical Note")

    analyze_btn.click(
        analyze_xray, 
        inputs=[img_input, name, age, gender], 
        outputs=[img_output, diagnosis_out, conf_out, report_out]
    )

# ==========================================
# 5. RENDER PRODUCTION DEPLOYMENT RUNNER
# ==========================================
if __name__ == "__main__":
    # Dynamically extract the environment port assigned by Render's routing interface
    server_port = int(os.environ.get("PORT", 7860))
    
    # Launch Gradio bound to 0.0.0.0 to expose it cleanly to Render's web proxy layer
    demo.launch(
        server_name="0.0.0.0", 
        server_port=server_port,
        prevent_thread_lock=False
    )
