<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f2027,100:0ea5e9&height=200&section=header&text=PNEUMONIA.AI&fontColor=ffffff&fontSize=40&fontAlignY=35&desc=Explainable%20AI%20for%20Pneumonia%20Detection%20from%20Chest%20X-Rays&descAlignY=55&descAlign=50"/>
</p>

<p align="center">
  <img src="https://skillicons.dev/icons?i=python,flask,html,css,javascript,git,github,vscode"/>
</p>

## 🧠 Intelligent AI Knowledge Retrieval System

## 🌟 Highlights

- **Explainable, not just accurate** — Grad-CAM heatmaps show exactly which lung regions drove each prediction.
- **Clinically-oriented pipeline** — Combines a ResNet50 classifier with an LLM-powered report generator for doctor-ready output.
- **End-to-end web experience** — Upload an X-ray, get a prediction, a visual explanation, and a downloadable PDF report.
- **Built for real clinical trust** — Designed around transparency and interpretability rather than black-box predictions.

## 📌 Overview

PNEUMONIA.AI is an intelligent medical imaging system designed to detect pneumonia from chest X-ray images using deep learning, with explainable AI outputs built in for clinical trust.

Unlike traditional black-box AI models, this system focuses on transparency, interpretability, and clinical usability. It not only predicts pneumonia but also shows *where* and *why* the model made that decision using Grad-CAM heatmaps, making it suitable for real-world healthcare environments.

## 🧠 Core Idea

> "AI should not just predict — it should explain."

This system combines:

- 🧠 Deep Learning (ResNet50 CNN)
- 🔍 Explainable AI (Grad-CAM)
- ☁️ Cloud LLM (Gemini 2.5 Flash)
- 📄 Automated clinical report generation
- 🌐 Web-based medical interface

## ⚙️ How It Works (Intelligent Pipeline)

<p align="center">
  <img width="350" height="930" alt="Pipeline diagram" src="https://github.com/user-attachments/assets/a4ef2b60-a575-41ad-941f-9f828c2f1710"/>
</p>

## 🧠 AI/ML Intelligence Layer

### Why ResNet50?

Deep networks suffer from vanishing gradients. ResNet solves this using residual learning:

```
H(x) = F(x) + x
```

- Enables deeper feature extraction
- Improves medical image understanding
- Captures fine lung abnormalities

### Explainable AI (Grad-CAM)

To ensure transparency in medical decisions, the model's attention is visualized directly on the X-ray:

<p align="center">
  <img width="261" height="168" alt="Grad-CAM visualization" src="https://github.com/user-attachments/assets/2bae5d2a-11bb-46d3-b635-d3ce37836a0d"/>
</p>

## 🔬 Why This Project Matters

**Traditional AI medical systems:**

- ❌ Black-box predictions
- ❌ No explanation
- ❌ Low clinical trust

**This system:**

- ✅ Transparent predictions
- ✅ Visual explanations
- ✅ Clinically interpretable output
- ✅ AI-generated report for doctors

## 🏗️ System Architecture

| Layer | Function |
|-------|----------|
| **Input Layer** | Chest X-ray image upload |
| **AI Layer** | ResNet50 classification model |
| **Explainability Layer** | Grad-CAM heatmaps |
| **LLM Layer** | Gemini 2.5 clinical report generator |
| **Output Layer** | Web UI + PDF report |

## 🛠️ Tech Stack

- 🐍 Python 3.11
- 🧠 TensorFlow / Keras
- 🔍 OpenCV
- 📊 NumPy, Matplotlib
- 🌐 Flask (Web App)
- ☁️ Gemini 2.5 Flash API
- 📄 ReportLab (PDF Generation)
- 📦 Scikit-learn

## 📊 Model Evaluation

The system is evaluated using:

- Accuracy
- Precision / Recall
- ROC Curve
- AUC Score

**Interpretation:**

- AUC → 1.0 = highly accurate model
- AUC → 0.5 = random prediction

**Grad-CAM output additionally:**

- Highlights infected lung regions
- Shows model reasoning visually
- Builds clinical trust

## 🏥 Real-World Impact

This system can assist:

- 🏥 Radiologists
- 🧑‍⚕️ Doctors in rural areas
- 🚑 Emergency diagnosis systems
- 📊 Medical research institutions

👉 It reduces diagnosis time and improves early detection of pneumonia.

## 📈 Future Improvements

- Multi-disease detection (TB, COVID-19)
- DICOM medical image support
- Mobile AI diagnostic app
- Cloud GPU deployment
- Hospital integration system

## 🖼️ Working Demo

<p align="center">
  <img width="1516" height="696" alt="Application screenshot" src="https://github.com/user-attachments/assets/253d091b-9b24-4632-b72b-3a4d85de3f01"/>
</p>

## 👨‍💻 Created & Maintained By

<p align="center">
  <img src="https://img.shields.io/badge/Author-Biswajit%20Pattanaik-0f2027?style=for-the-badge&logo=github&logoColor=white" alt="Author Badge"/>
</p>

<p align="center">
  <b>Biswajit Pattanaik</b><br/>
  <i>B.Tech (ECE) | AI & Embedded Systems Developer | Centurion University</i>
</p>

<p align="center">
  ⭐ If this project helped you, consider giving it a star — it goes a long way!<br/>
  🐛 Found a bug or have an idea? Issues and pull requests are always welcome.<br/>
  🤝 Open to feedback, collaboration, and discussion.
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:0ea5e9,100:0f2027&height=100&section=footer"/>
</p>
