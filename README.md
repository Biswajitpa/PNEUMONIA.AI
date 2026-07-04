<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f2027,100:0ea5e9&height=200&section=header&text=PNEUMONIA.AI&fontColor=ffffff&fontSize=40&fontAlignY=35&desc=Explainable%20AI%20for%20Pneumonia%20Detection%20from%20Chest%20X-Rays&descAlignY=55&descAlign=50"/>
</p>

<p align="center">
  <img src="https://skillicons.dev/icons?i=python,flask,html,css,javascript,git,github,vscode"/>
</p>

<p align="center">
  <b>"AI should not just predict — it should explain."</b>
</p>

## 📌 What It Does

PNEUMONIA.AI detects pneumonia from chest X-rays using a **ResNet50 CNN**, then explains *why* with **Grad-CAM heatmaps** — turning a black-box prediction into a visual, doctor-readable diagnosis. A built-in LLM (Gemini 2.5 Flash) auto-generates a clinical report as a downloadable PDF.

## ⚡ Why It Stands Out

| Typical Medical AI | PNEUMONIA.AI |
|---|---|
| ❌ Black-box prediction | ✅ Visual explanation via Grad-CAM |
| ❌ No clinical context | ✅ Auto-generated clinical report |
| ❌ Hard to trust | ✅ Built for interpretability & trust |

## ⚙️ Pipeline

**X-ray upload → ResNet50 classification → Grad-CAM heatmap → Gemini clinical report → Web UI + PDF**

<p align="center">
  <img width="280" alt="Pipeline diagram" src="https://github.com/user-attachments/assets/a4ef2b60-a575-41ad-941f-9f828c2f1710"/>
  <img width="220" alt="Grad-CAM visualization" src="https://github.com/user-attachments/assets/2bae5d2a-11bb-46d3-b635-d3ce37836a0d"/>
</p>

## 🛠️ Tech Stack

`Python 3.11` · `TensorFlow/Keras` · `OpenCV` · `Flask` · `Gemini 2.5 Flash` · `ReportLab` · `Scikit-learn`

## 📊 Evaluated On

Accuracy · Precision/Recall · ROC-AUC — with Grad-CAM confirming the model attends to the actual infected lung regions, not spurious patterns.

## 🏥 Impact

Assists radiologists, rural clinics, and emergency triage by cutting diagnosis time and flagging pneumonia early — with reasoning a doctor can actually verify.

## 📈 What's Next

Multi-disease detection (TB, COVID-19) · DICOM support · Mobile app · Cloud GPU deployment · Hospital system integration

## 🖼️ Demo

<p align="center">
  <img width="800" alt="Application screenshot" src="https://github.com/user-attachments/assets/253d091b-9b24-4632-b72b-3a4d85de3f01"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Biswajit%20Pattanaik-0f2027?style=for-the-badge&logo=github&logoColor=white" alt="Author Badge"/>
</p>

<p align="center">
  <b>Biswajit Pattanaik</b> — B.Tech (ECE), AI & Embedded Systems Developer, Centurion University<br/>
  ⭐ Star it if you find it useful · 🤝 Issues & PRs welcome
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:0ea5e9,100:0f2027&height=100&section=footer"/>
</p>
