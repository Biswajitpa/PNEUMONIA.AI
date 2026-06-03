## 🩺 PNEUMONIA.AI

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f2027,100:0ea5e9&height=200&text=PNEUMONIA.AI&fontColor=ffffff&fontSize=40"/>
</p>

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?lines=AI+Powered+Medical+Diagnosis+System;Explainable+Deep+Learning+for+Healthcare;ResNet50+%2B+Grad-CAM+XAI+Framework"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Domain-Medical%20AI-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Model-ResNet50-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/XAI-GradCAM-orange?style=for-the-badge"/>
</p>

## 📌 Overview
PNEUMONIA.AI is an intelligent medical imaging system designed to detect pneumonia from chest X-ray images using deep learning and provide explainable AI outputs for clinical trust.
Unlike traditional black-box AI models, this system focuses on transparency, interpretability, and clinical usability.It not only predicts pneumonia but also shows where and why the model made that decision using Grad-CAM heatmaps, making it suitable for real-world healthcare environments.

## 🧠Core Idea
- “AI should not just predict — it should explain.”
    - This system combines:
    - 🧠 Deep Learning (ResNet50 CNN)
    - 🔍 Explainable AI (Grad-CAM)
    - ☁️ Cloud LLM (Gemini 2.5 Flash)
    - 📄 Automated Clinical Report Generation
    - 🌐 Web-based Medical Interface
## ⚙️ How It Works (Intelligent Pipeline)
<img width="350" height="930" alt="diagram-export-6-4-2026-12_20_13-AM" src="https://github.com/user-attachments/assets/a4ef2b60-a575-41ad-941f-9f828c2f1710" />

## 🧠 AI/ML Intelligence Layer
- Why ResNet50?
- Deep networks suffer from vanishing gradients. ResNet solves this using residual learning:
   - H(x)=F(x)+x
   - Enables deeper feature extraction
   - Improves medical image understanding
   - Captures fine lung abnormalities
- Explainable AI (Grad-CAM)
   -To ensure transparency in medical decisions:
<img width="261" height="168" alt="image" src="https://github.com/user-attachments/assets/2bae5d2a-11bb-46d3-b635-d3ce37836a0d" />
## 🔬 Why This Project Matters
- Traditional AI medical systems:
- ❌ Black-box predictions
- ❌ No explanation
- ❌ Low clinical trust

## This system:

-  Transparent predictions
- Visual explanations
- Clinically interpretable output
- AI-generated report for doctors
## 🏗️ System Architecture
Input Layer: Chest X-ray Image Upload
AI Layer: ResNet50 Classification Model
Explainability Layer: Grad-CAM Heatmaps
LLM Layer: Gemini 2.5 Clinical Report Generator
Output Layer: Web UI + PDF Report
🛠️ Tech Stack
🐍 Python 3.11
🧠 TensorFlow / Keras
🔍 OpenCV
📊 NumPy, Matplotlib
🌐 Flask (Web App)
☁️ Gemini 2.5 Flash API
📄 ReportLab (PDF Generation)
📦 Scikit-learn
📊 Model Evaluation

The system is evaluated using:
Accuracy
Precision / Recall
ROC Curve
AUC Score
Interpretation:
AUC → 1.0 = highly accurate model
AUC → 0.5 = random prediction
-  Highlights infected lung regions
-  hows model reasoning visually
-  uilds clinical trust

