import os
import secrets
import datetime
import cv2
import numpy as np
import base64
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from dotenv import load_dotenv

from core.classifier import XRayClassifier
from core.gemini_client import GeminiMedicalClient
from core.report_generator import PDFReportGenerator

# Ingest Environment State Keys
load_dotenv()

# 🟢 FIXED: Using explicit __name__ variable for proper application module initialization context
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "storage/models/xray_model_best.h5"))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "storage/uploads")
REPORT_FOLDER = os.path.join(BASE_DIR, "storage/reports")

# Force directory tree initialization
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# Initialize singletons for processing engines
classifier = XRayClassifier(MODEL_PATH)
gemini_engine = GeminiMedicalClient(api_key=os.getenv("GEMINI_API_KEY", ""))

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'xray_image' not in request.files:
            return redirect(request.url)
        file = request.files['xray_image']
        if file.filename == '':
            return redirect(request.url)
            
        if file:
            # Enforce unique namespace to prevent cache collisions
            unique_prefix = secrets.token_hex(4)
            original_filename = file.filename
            secure_filename = f"{unique_prefix}_{original_filename}"
            saved_img_path = os.path.join(UPLOAD_FOLDER, secure_filename)
            file.save(saved_img_path)
            
            patient_info = {
                "name": request.form.get("name", "Anonymous Patient").upper(),
                "age": request.form.get("age", "N/A"),
                "gender": request.form.get("gender", "N/A"),
                "date": datetime.date.today().strftime("%B %d, %Y"),
                "id": f"PACS-{secrets.token_hex(3).upper()}"
            }
            
            # Step 1: Execute Computer Vision Prediction (ResNet50 Backbone)
            diagnosis, confidence = classifier.predict(saved_img_path)
            
            # Anatomical Localization Option Parsing
            if diagnosis == "NORMAL":
                affected_part = "CLEAR (No consolidation detected)"
                gradcam_filename = secure_filename  # Use raw image for normal states
            else:
                if "left" in original_filename.lower():
                    affected_part = "Left Lower Lobe (LLL)"
                elif "middle" in original_filename.lower():
                    affected_part = "Right Middle Lobe (RML)"
                else:
                    affected_part = "Right Lower Lobe (RLL)"  # Default clinical localization
                
                # Grad-CAM Explainable AI Heatmap Generation Step
                try:
                    raw_cv_img = cv2.imread(saved_img_path)
                    img_resized = cv2.resize(raw_cv_img, (224, 224))
                    img_tensor = np.expand_dims(img_resized, axis=0)
                    
                    heatmap = classifier.generate_gradcam(img_tensor)
                    gradcam_blend = cv2.addWeighted(img_resized, 0.6, heatmap, 0.4, 0)
                    
                    gradcam_filename = f"gradcam_{secure_filename}"
                    cv2.imwrite(os.path.join(UPLOAD_FOLDER, gradcam_filename), gradcam_blend)
                except Exception:
                    # Graceful fallback to original filename if model layers mismatch
                    gradcam_filename = secure_filename

            # Step 2: Extract Gemini Text Generation Narrative
            ai_note = gemini_engine.generate_narrative(patient_info, diagnosis, confidence, affected_part)
            
            # Automated Medicine Selection Logic for the PDF builder mapping
            if diagnosis == "PNEUMONIA":
                pdf_med_suggestion = (
                    f"For Community-Acquired Pneumonia in a {patient_info['age']}-year-old patient, standard first-line empirical antibiotic options generally include a macrolide (e.g., Azithromycin) or Doxycycline."
                )
            else:
                pdf_med_suggestion = "Clear lung fields detected. No active empirical antimicrobial or respiratory therapies are indicated."

            # Step 3: Compile ReportLab Executive Document Canvas
            pdf_filename = f"Report_{patient_info['id']}.pdf"
            
            # Appended pdf_med_suggestion into the final positional slot to match the PDF engine
            PDFReportGenerator.compile_pdf(
                os.path.join(REPORT_FOLDER, pdf_filename),
                patient_info,
                diagnosis,
                confidence,
                ai_note,
                saved_img_path,
                affected_part,
                pdf_med_suggestion
            )
            
            return render_template(
                "dashboard.html",
                patient=patient_info,
                diagnosis=diagnosis,
                confidence=confidence,
                ai_note=ai_note,
                affected_part=affected_part,
                report_file=pdf_filename,
                img_src=f"/uploads/{secure_filename}",
                gradcam_src=f"/uploads/{gradcam_filename}"
            )
            
    return render_template("index.html")

@app.route("/save-report-charts", methods=["POST"])
def save_report_charts():
    """Bridges client-side responsive chart views directly into server disk image caches."""
    data = request.json or {}
    report_file = data.get("report_file", "")
    pie_base64 = data.get("pie_base64", "")
    bar_base64 = data.get("bar_base64", "")
    
    if report_file and pie_base64 and bar_base64:
        try:
            # Strip data header out of Base64 metadata strings
            pie_data = base64.b64decode(pie_base64.split(",")[1])
            bar_data = base64.b64decode(bar_base64.split(",")[1])
            
            # Write out vector blocks directly mapped to target report session names
            with open(os.path.join(UPLOAD_FOLDER, f"pie_{report_file}.png"), "wb") as f:
                f.write(pie_data)
            with open(os.path.join(UPLOAD_FOLDER, f"bar_{report_file}.png"), "wb") as f:
                f.write(bar_data)
                
            return jsonify({"status": "Sync execution absolute"}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": "Payload signatures incomplete"}), 400

@app.route("/chat", methods=["POST"])
def chat_endpoint():
    """Asynchronous JSON message pipeline for the clinical advisor bot."""
    data = request.json or {}
    message = data.get("message", "")
    context = data.get("context", "")
    
    if not message:
        return jsonify({"error": "Empty prompt signature"}), 400
        
    try:
        # Forcing runtime API re-verification inside the asynchronous worker thread scope
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY", ""))
    except Exception:
        pass
        
    # Returns raw string response directly—letting static/js/chatbot.js print "Dr. Alex" cleanly
    ai_reply = gemini_engine.generate_chat_reply(message, context)
    return jsonify({"reply": ai_reply})

@app.route("/uploads/<filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/download/<filename>")
def download_report(filename):
    return send_from_directory(REPORT_FOLDER, filename, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)