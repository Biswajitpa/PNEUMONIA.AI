import os
import requests
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from core.classifier import XRayClassifier
from core.gemini_client import GeminiMedicalClient

# Set page configuration
st.set_page_config(page_title="PNEUMONIA.AI", page_icon="🧠", layout="wide")

# ==========================================
# 1. PATH CONFIGURATION & FILE DOWNLOADER
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "storage", "models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "xray_model_best.h5")

FILE_ID = "1-XmVDb3ldcpbMd--OX_PkbzBkDgceI0q"
MODEL_URL = f"https://docs.google.com/uc?export=download&id={FILE_ID}&confirm=t"

@st.cache_resource
def load_medical_engines():
    """Downloads and caches models so they only load once into memory."""
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading ResNet50 model weights from Google Drive (139MB)..."):
            try:
                response = requests.get(MODEL_URL, stream=True)
                response.raise_for_status()
                with open(MODEL_PATH, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                st.success("✅ Model weights downloaded successfully!")
            except Exception as e:
                st.error(f"❌ Cloud download failed: {e}")
                return None, None

    classifier = XRayClassifier(MODEL_PATH)
    gemini_engine = GeminiMedicalClient(api_key=os.getenv("GEMINI_API_KEY"))
    return classifier, gemini_engine

# Initialize engines cleanly
classifier, gemini_engine = load_medical_engines()

# ==========================================
# 2. STATE MANAGEMENT (Prevents UI Disappearing)
# ==========================================
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

# ==========================================
# 3. UI LAYOUT & USER INPUTS
# ==========================================
st.title("🧠 PNEUMONIA.AI - Clinical Diagnostic Support")
st.markdown("Upload a chest X-ray to detect pneumonia and view the explainable heatmap.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Patient Information & Input")
    uploaded_file = st.file_uploader("Upload Chest X-Ray", type=["jpg", "jpeg", "png"])
    
    name = st.text_input("Patient Name")
    age = st.number_input("Patient Age", min_value=0, max_value=120, value=30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    
    run_analysis = st.button("Run Analysis", type="primary")

    if run_analysis:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            with st.spinner("Analyzing X-ray scan..."):
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
                
                # Save into persistent state session memory
                st.session_state.analysis_results = {
                    "output_img": output_img,
                    "diagnosis": diagnosis,
                    "confidence": f"{confidence:.2f}%",
                    "ai_note": ai_note
                }
        else:
            st.warning("⚠️ Please upload a chest X-ray scan first.")

with col2:
    st.subheader("📊 Diagnostic Output")
    
    # Clean render boundary path
    if st.session_state.analysis_results is not None:
        results = st.session_state.analysis_results
        
        # Render the components continuously even when user tweaks left inputs
        st.image(results["output_img"], caption="Explainable Heatmap / Scan Matrix", use_container_width=True)
        st.metric(label="Model Diagnosis", value=str(results["diagnosis"]))
        st.metric(label="Confidence Score", value=results["confidence"])
        
        st.text_area("AI Clinical Note", value=results["ai_note"], height=200)
    else:
        st.info("💡 Awaiting input scan analysis parameters. Upload an X-ray and click 'Run Analysis'.")
