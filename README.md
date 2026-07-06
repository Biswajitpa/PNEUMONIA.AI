<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f2027,100:0ea5e9&height=180&section=header&text=PNEUMONIA.AI&fontColor=ffffff&fontSize=38&fontAlignY=35&desc=Explainable%20AI%20for%20Pneumonia%20Detection%20from%20Chest%20X-Rays&descAlignY=55&descAlign=50"/>
</p>

# PNEUMONIA.AI

**Explainable AI for Pneumonia Detection from Chest X-Rays**

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web%20App-000000?style=flat-square&logo=flask&logoColor=white)
![Status](https://img.shields.io/badge/Status-Production--Ready-2ea44f?style=flat-square)
![License](https://img.shields.io/badge/Status-Active-0ea5e9?style=flat-square)

<p align="center">
  <img src="https://skillicons.dev/icons?i=python,tensorflow,flask,opencv,html,css,javascript,git,github,vscode"/>
</p>

---

## Overview

PNEUMONIA.AI is a medical imaging system that detects pneumonia from chest X-rays using deep learning, with explainability built into its core rather than added as an afterthought.

Instead of returning a single black-box prediction, the system shows **where** in the lung the model is focusing and **why**, using Grad-CAM heatmaps, and generates a doctor-readable clinical report from the result — reducing diagnosis time and improving clinical trust in the output.

**Design principle:** *AI should not just predict — it should explain.*

---

## Key Capabilities

- **Explainable, not just accurate** — Grad-CAM heatmaps localize the lung regions driving each prediction.
- **Clinically-oriented pipeline** — A ResNet50 classifier is paired with an LLM-based report generator to produce structured, doctor-ready summaries.
- **End-to-end web experience** — Upload an X-ray → get a prediction → get a visual explanation → download a PDF report.
- **Built for clinical trust** — Every prediction is paired with visual and textual justification instead of a bare label.

---

## System Design

### Architecture

```mermaid
flowchart TD
    U[👤 User / Patient-Doctor]
    WEB[💻 Flask Web App]
    PRE[⚙️ Preprocessing<br/>Resize · Normalize]
    CNN[🧠 ResNet50 Classifier]
    GC[🔍 Grad-CAM Engine]
    GEMINI[☁️ Gemini 2.5 Flash]
    PDF[📄 ReportLab PDF]
    DASH[📊 Results Dashboard]

    U -->|Upload X-ray| WEB
    WEB -->|Send image| PRE
    PRE -->|Tensor input| CNN
    CNN -->|Prediction + Confidence| GEMINI
    CNN -->|Activation maps| GC
    GC -->|Heatmap overlay| GEMINI
    GC -->|Heatmap overlay| DASH
    GEMINI -->|Clinical narrative| PDF
    GEMINI -->|Clinical narrative| DASH
    PDF -->|Download report| U
    DASH -->|Render results| WEB
    WEB -->|Serve response| U

    classDef user fill:#22d3ee,stroke:#0f2027,color:#0f2027,stroke-width:2px
    classDef web fill:#38bdf8,stroke:#0f2027,color:#0f2027,stroke-width:2px
    classDef prep fill:#a3e635,stroke:#0f2027,color:#0f2027,stroke-width:2px
    classDef cnn fill:#facc15,stroke:#0f2027,color:#0f2027,stroke-width:2px
    classDef gc fill:#ec4899,stroke:#0f2027,color:#ffffff,stroke-width:2px
    classDef gemini fill:#0f172a,stroke:#22d3ee,color:#ffffff,stroke-width:2px
    classDef pdf fill:#f97316,stroke:#0f2027,color:#0f2027,stroke-width:2px
    classDef dash fill:#34d399,stroke:#0f2027,color:#0f2027,stroke-width:2px

    class U user
    class WEB web
    class PRE prep
    class CNN cnn
    class GC gc
    class GEMINI gemini
    class PDF pdf
    class DASH dash
```

### Request Flow (Sequence)

```mermaid
sequenceDiagram
    participant V as 🧑 User
    participant B as ⚙️ Flask Backend
    participant M as 🧠 ResNet50 Model
    participant G as 🔍 Grad-CAM
    participant L as ☁️ Gemini 2.5 Flash
    participant R as 📄 ReportLab

    V->>B: Upload chest X-ray
    B->>M: Preprocessed image tensor
    M-->>B: Prediction + confidence score
    B->>G: Request activation heatmap
    G-->>B: Grad-CAM overlay image
    B->>L: Prediction + heatmap context
    L-->>B: Structured clinical report
    B->>R: Compile report + images
    R-->>B: Generated PDF
    B-->>V: Real-time result + downloadable PDF
    Note over V,R: On new upload → pipeline resets and repeats
```

**Data flow, step by step:**

1. **Upload** - The user submits a chest X-ray through the Flask web interface.
2. **Preprocess** - The image is resized, normalized, and optionally augmented for consistent model input.
3. **Classify** - ResNet50 predicts *Pneumonia* or *Normal* with an associated confidence score.
4. **Explain** - Grad-CAM back-propagates gradients from the final convolutional layer to produce a heatmap, which is overlaid on the original X-ray.
5. **Reason** - The prediction, confidence score, and heatmap context are passed to Gemini 2.5 Flash, which drafts a structured clinical narrative.
6. **Deliver** - The dashboard renders the prediction, heatmap, and report inline, while ReportLab packages everything into a downloadable PDF.

### Component Breakdown

| Layer | Responsibility | Key Technology |
|---|---|---|
| **Input Layer** | Accepts and validates chest X-ray uploads | Flask, OpenCV |
| **Preprocessing Layer** | Resizes, normalizes, and augments images for inference | OpenCV, NumPy |
| **Classification Layer** | Predicts pneumonia vs. normal from image features | ResNet50 (TensorFlow/Keras) |
| **Explainability Layer** | Generates Grad-CAM attention maps over the input image | TensorFlow, OpenCV, Matplotlib |
| **Reporting Layer** | Converts prediction + visual evidence into a clinical narrative | Gemini 2.5 Flash API |
| **Output Layer** | Renders results in-browser and as a downloadable report | Flask, ReportLab |

### Why ResNet50

Deep CNNs are prone to vanishing gradients as depth increases. ResNet50 mitigates this through residual learning, which lets the network skip a layer's transformation when it isn't useful:

```
H(x) = F(x) + x
```

This residual connection allows the network to:
- Train deeper feature extractors reliably
- Learn subtler patterns in lung tissue
- Capture fine-grained abnormalities that shallower CNNs miss

### Why Grad-CAM

Grad-CAM (Gradient-weighted Class Activation Mapping) traces the classifier's gradients back to the final convolutional layer to produce a heatmap of "where the model looked." This turns a single prediction into a visually verifiable claim — a radiologist can immediately see whether the model's attention aligns with the actual region of concern.

---

## Why This Approach Matters

| Traditional black-box systems | PNEUMONIA.AI |
|---|---|
| Opaque predictions | Transparent, traceable predictions |
| No visual justification | Grad-CAM heatmap on every result |
| Difficult to audit clinically | Interpretable, review-ready output |
| Raw label only | Structured, AI-generated clinical report |

---

## Tech Stack

- **Language:** Python 3.11
- **Modeling:** TensorFlow / Keras (ResNet50)
- **Explainability:** OpenCV, Grad-CAM
- **Data / Numerics:** NumPy, Matplotlib, Scikit-learn
- **Backend:** Flask
- **LLM Reporting:** Gemini 2.5 Flash API
- **Document Generation:** ReportLab (PDF)

---

## Model Evaluation

Performance is assessed using standard classification metrics:

- Accuracy
- Precision / Recall
- ROC Curve
- AUC Score

**Interpretation guide:**
- AUC → 1.0 indicates a highly discriminative model
- AUC → 0.5 indicates performance no better than random guessing

Grad-CAM outputs are evaluated qualitatively alongside these metrics — a well-performing model should localize attention over clinically relevant lung regions, not incidental image artifacts.

---

## Real-World Applications

- Radiology departments seeking a second-opinion / triage tool
- Doctors in under-resourced or rural settings without on-site radiologists
- Emergency diagnostic workflows requiring rapid triage
- Medical research institutions studying interpretable diagnostic AI

---

## Roadmap

- [ ] Multi-disease detection (tuberculosis, COVID-19)
- [ ] DICOM medical image format support
- [ ] Mobile diagnostic application
- [ ] Cloud GPU deployment for scalable inference
- [ ] Hospital information system (HIS) integration

---

## About the Author

<p align="center">
  <img src="https://img.shields.io/badge/Author-Biswajit%20Pattanaik-0f2027?style=for-the-badge&logo=github&logoColor=white"/>
</p>

**Biswajit Pattanaik**
B.Tech, Electronics & Communication Engineering — Centurion University
AI & Embedded Systems Developer

Biswajit designed and built PNEUMONIA.AI end-to-end — from data preprocessing and model training, through the Grad-CAM explainability pipeline, to the Flask web interface and automated PDF reporting. The project reflects a deliberate design philosophy: an AI system intended for clinical use should never be a black box, and every prediction it makes should come with visual and written evidence a doctor can independently verify.

**Connect / Contribute:**
- ⭐ If this project helped you, a star means a lot and helps others discover it.
- 🐛 Found a bug or have an improvement in mind? Open an issue.
- 🤝 Interested in collaborating or extending the project? Pull requests and discussions are welcome.

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:0ea5e9,100:0f2027&height=100&section=footer"/>
</p>
