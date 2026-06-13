import os
import gradio as gr
import numpy as np
import cv2
from PIL import Image
from core.classifier import XRayClassifier
from core.gemini_client import GeminiMedicalClient
from core.report_generator import PDFReportGenerator

# 1. Initialize Engine (Use environment variables for paths)
MODEL_PATH = os.getenv("MODEL_PATH", "storage/models/xray_model_best.h5")
classifier = XRayClassifier(MODEL_PATH)
gemini_engine = GeminiMedicalClient(api_key=os.getenv("GEMINI_API_KEY"))

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

# 2. UI Layout using Gradio Blocks
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

if __name__ == "__main__":
    demo.launch()
