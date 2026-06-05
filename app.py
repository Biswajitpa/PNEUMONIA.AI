import os
import secrets
import datetime
import numpy as np
import base64

from flask import Flask, render_template, request, redirect, send_from_directory, jsonify
from dotenv import load_dotenv

from core.classifier import XRayClassifier
from core.gemini_client import GeminiMedicalClient
from core.report_generator import PDFReportGenerator

# Load environment variables
load_dotenv()

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.join(BASE_DIR, "storage/models/xray_model_best.h5")
)

UPLOAD_FOLDER = os.path.join(BASE_DIR, "storage/uploads")
REPORT_FOLDER = os.path.join(BASE_DIR, "storage/reports")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# Initialize AI engines
classifier = XRayClassifier(MODEL_PATH)
gemini_engine = GeminiMedicalClient(api_key=os.getenv("GEMINI_API_KEY", ""))


# ---------------- HOME ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        if "xray_image" not in request.files:
            return redirect(request.url)

        file = request.files["xray_image"]

        if file.filename == "":
            return redirect(request.url)

        # Save image safely
        unique_prefix = secrets.token_hex(4)
        original_filename = file.filename
        secure_filename = f"{unique_prefix}_{original_filename}"

        saved_img_path = os.path.join(UPLOAD_FOLDER, secure_filename)
        file.save(saved_img_path)

        # Patient info
        patient_info = {
            "name": request.form.get("name", "Anonymous Patient").upper(),
            "age": request.form.get("age", "N/A"),
            "gender": request.form.get("gender", "N/A"),
            "date": datetime.date.today().strftime("%B %d, %Y"),
            "id": f"PACS-{secrets.token_hex(3).upper()}"
        }

        # ---------------- STEP 1: ML Prediction ----------------
        diagnosis, confidence = classifier.predict(saved_img_path)

        # ---------------- Affected Part Logic ----------------
        if diagnosis == "NORMAL":
            affected_part = "CLEAR (No consolidation detected)"
            gradcam_filename = secure_filename

        else:
            if "left" in original_filename.lower():
                affected_part = "Left Lower Lobe (LLL)"
            elif "middle" in original_filename.lower():
                affected_part = "Right Middle Lobe (RML)"
            else:
                affected_part = "Right Lower Lobe (RLL)"

            # ---------------- STEP 2: Grad-CAM (SAFE IMPORT) ----------------
            try:
                import cv2  # ✅ LOCAL IMPORT (FIX FOR VERCEL)

                raw_cv_img = cv2.imread(saved_img_path)

                if raw_cv_img is not None:
                    img_resized = cv2.resize(raw_cv_img, (224, 224))
                    img_tensor = np.expand_dims(img_resized, axis=0)

                    heatmap = classifier.generate_gradcam(img_tensor)
                    gradcam_blend = cv2.addWeighted(img_resized, 0.6, heatmap, 0.4, 0)

                    gradcam_filename = f"gradcam_{secure_filename}"
                    cv2.imwrite(os.path.join(UPLOAD_FOLDER, gradcam_filename), gradcam_blend)
                else:
                    gradcam_filename = secure_filename

            except Exception:
                gradcam_filename = secure_filename

        # ---------------- STEP 3: Gemini AI Report ----------------
        ai_note = gemini_engine.generate_narrative(
            patient_info,
            diagnosis,
            confidence,
            affected_part
        )

        # ---------------- STEP 4: Medicine Suggestion ----------------
        if diagnosis == "PNEUMONIA":
            pdf_med_suggestion = (
                f"For Community-Acquired Pneumonia in a {patient_info['age']}-year-old patient, "
                "first-line treatment may include Azithromycin or Doxycycline."
            )
        else:
            pdf_med_suggestion = "No active infection detected. No treatment required."

        # ---------------- STEP 5: PDF Report ----------------
        pdf_filename = f"Report_{patient_info['id']}.pdf"

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


# ---------------- SAVE CHARTS ----------------
@app.route("/save-report-charts", methods=["POST"])
def save_report_charts():
    data = request.json or {}

    report_file = data.get("report_file", "")
    pie_base64 = data.get("pie_base64", "")
    bar_base64 = data.get("bar_base64", "")

    if report_file and pie_base64 and bar_base64:
        try:
            pie_data = base64.b64decode(pie_base64.split(",")[1])
            bar_data = base64.b64decode(bar_base64.split(",")[1])

            with open(os.path.join(UPLOAD_FOLDER, f"pie_{report_file}.png"), "wb") as f:
                f.write(pie_data)

            with open(os.path.join(UPLOAD_FOLDER, f"bar_{report_file}.png"), "wb") as f:
                f.write(bar_data)

            return jsonify({"status": "success"}), 200

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Invalid payload"}), 400


# ---------------- CHAT API ----------------
@app.route("/chat", methods=["POST"])
def chat_endpoint():
    data = request.json or {}

    message = data.get("message", "")
    context = data.get("context", "")

    if not message:
        return jsonify({"error": "Empty message"}), 400

    ai_reply = gemini_engine.generate_chat_reply(message, context)

    return jsonify({"reply": ai_reply})


# ---------------- FILE SERVING ----------------
@app.route("/uploads/<filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/download/<filename>")
def download_report(filename):
    return send_from_directory(REPORT_FOLDER, filename, as_attachment=True)


# ---------------- MAIN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
