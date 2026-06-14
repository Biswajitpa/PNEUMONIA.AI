import io
import textwrap
import os
import base64
import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import matplotlib
matplotlib.use('Agg') # Prevents GUI terminal errors on headless cloud servers
import matplotlib.pyplot as plt
from PIL import Image

# 🚨 TENSORFLOW HAS BEEN REMOVED FROM THIS BLOCK TO PREVENT THE CONTAINER CRASH
# Your core.classifier module will now handle the model weights via onnxruntime.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
from dotenv import load_dotenv


from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, Image as RLImage,
                                 HRFlowable)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT

import requests as _requests
# ============================================================
# AUTOMATIC ONNX MODEL WEIGHTS RUNTIME DOWNLOADER
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "storage", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "xray_model_best.onnx")

# Create the folder structure automatically inside the cloud container if missing
os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    with st.spinner("📥 Synchronizing optimized ONNX model weights from cloud storage... please wait."):
        # Pulls directly from your shared Google Drive file link
        FILE_ID = "1-XmVDb3ldcpbMd--OX_PkbzBkDgceI0q"
        MODEL_URL = f"https://docs.google.com/uc?export=download&id={FILE_ID}"
        
        try:
            # Note: Using your imported _requests alias safely
            response = _requests.get(MODEL_URL, stream=True)
            response.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            st.success("✅ Model weights completely synchronized successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Cloud model synchronization failed: {e}")

load_dotenv()

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="PNEUMONIA.AI v4.1", page_icon="🩺", layout="wide")

# ============================================================
# GLOBAL CSS
# ============================================================
st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(circle at 15% 0%,  rgba(45,212,191,.10) 0%, transparent 45%),
        radial-gradient(circle at 85% 15%, rgba(5,150,105,.10)  0%, transparent 50%),
        radial-gradient(circle at 50% 100%,rgba(13,148,136,.08) 0%, transparent 55%),
        #03100c !important;
    color:#fff !important;
    background-attachment:fixed;
}
.amb{position:fixed;border-radius:50%;filter:blur(80px);opacity:.35;z-index:0;pointer-events:none;animation:orbF 18s ease-in-out infinite;}
.amb1{width:380px;height:380px;top:-120px;left:-100px;background:radial-gradient(circle,#2dd4bf 0%,transparent 70%);animation-delay:0s;}
.amb2{width:420px;height:420px;top:30%;right:-150px;background:radial-gradient(circle,#059669 0%,transparent 70%);animation-delay:-6s;}
.amb3{width:320px;height:320px;bottom:-100px;left:30%;background:radial-gradient(circle,#14b8a6 0%,transparent 70%);animation-delay:-12s;}
@keyframes orbF{0%,100%{transform:translate(0,0) scale(1)}33%{transform:translate(40px,30px) scale(1.08)}66%{transform:translate(-30px,-20px) scale(.95)}}

.topbar{display:flex;justify-content:space-between;align-items:center;
    background:rgba(13,148,136,.06);backdrop-filter:blur(18px);
    padding:14px 28px;border:1px solid rgba(45,212,191,.14);border-radius:14px;
    margin-top:-50px;margin-bottom:36px;
    box-shadow:0 8px 32px rgba(0,0,0,.35),inset 0 1px 0 rgba(255,255,255,.04);
    animation:fadeUp .6s ease both;position:relative;overflow:hidden;}
.topbar::after{content:"";position:absolute;top:0;left:-40%;width:40%;height:100%;
    background:linear-gradient(100deg,transparent,rgba(45,212,191,.08),transparent);
    animation:sweep 6s ease-in-out infinite;}
@keyframes sweep{0%{left:-40%}50%{left:110%}100%{left:110%}}

div[data-testid="stVerticalBlockBorderWrapper"]>div{
    background:linear-gradient(165deg,rgba(45,212,191,.07) 0%,rgba(13,148,136,.04) 100%) !important;
    backdrop-filter:blur(20px) !important;border-radius:18px !important;
    box-shadow:0 16px 48px rgba(0,0,0,.45),inset 0 1px 0 rgba(255,255,255,.06) !important;
    transition:all .45s cubic-bezier(.22,1,.36,1) !important;}
div[data-testid="stVerticalBlockBorderWrapper"]{border-color:rgba(45,212,191,.16) !important;border-radius:18px !important;animation:fadeUp .7s ease both;}
div[data-testid="stVerticalBlockBorderWrapper"]:hover>div{
    box-shadow:0 20px 60px rgba(13,148,136,.22),0 0 0 1px rgba(45,212,191,.18),inset 0 1px 0 rgba(255,255,255,.08) !important;
    transform:translateY(-3px);}

.pform div[data-testid="stVerticalBlockBorderWrapper"]{max-width:700px;margin:0 auto;border-color:rgba(45,212,191,.22) !important;}
.pform div[data-testid="stVerticalBlockBorderWrapper"]>div{
    background:linear-gradient(165deg,rgba(45,212,191,.10) 0%,rgba(5,150,105,.04) 100%) !important;
    box-shadow:0 24px 60px rgba(0,0,0,.6),inset 0 1px 0 rgba(255,255,255,.08),0 0 80px rgba(45,212,191,.06) !important;}
.pform div[data-testid="stVerticalBlockBorderWrapper"]:hover>div{transform:none;}

.flabel-g{font-family:'Courier New',monospace;font-size:11px;font-weight:bold;
    color:#6ee7b7 !important;letter-spacing:1px;margin-bottom:6px;margin-top:18px;text-transform:uppercase;}
.flabel-t{font-family:'Courier New',monospace;font-size:11px;font-weight:bold;
    color:#2dd4bf !important;letter-spacing:1px;margin-bottom:6px;margin-top:18px;text-transform:uppercase;}
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] div[data-baseweb="select"]{
    background-color:rgba(8,17,13,.6) !important;border:1px solid #1e3a2f !important;
    color:#e2f8f0 !important;border-radius:10px !important;padding:10px !important;transition:all .3s ease !important;}
div[data-testid="stTextInput"] input:focus,div[data-testid="stNumberInput"] input:focus{
    border-color:#2dd4bf !important;box-shadow:0 0 0 3px rgba(45,212,191,.18) !important;}

div.stButton>button{
    background:rgba(13,148,136,.08) !important;color:#6ee7b7 !important;
    font-family:sans-serif !important;font-weight:600 !important;font-size:13px !important;
    border:1px solid rgba(45,212,191,.3) !important;width:100% !important;
    padding:14px 0 !important;border-radius:10px !important;
    transition:all .35s cubic-bezier(.22,1,.36,1) !important;}
div.stButton>button:hover{background:rgba(45,212,191,.14) !important;border-color:#2dd4bf !important;color:#fff !important;transform:translateY(-2px);}
div.stButton>button[kind="primary"]{
    background:linear-gradient(90deg,#2dd4bf 0%,#059669 100%) !important;
    color:#03100c !important;font-family:'Courier New',monospace !important;font-weight:bold !important;
    font-size:13px !important;border:none !important;text-transform:uppercase;letter-spacing:1px;
    box-shadow:0 4px 24px rgba(5,150,105,.4) !important;margin-top:30px;}
div.stButton>button[kind="primary"]:hover{transform:translateY(-3px) scale(1.01);box-shadow:0 10px 32px rgba(45,212,191,.55) !important;}
div[data-testid="stDownloadButton"]>button{
    background:linear-gradient(90deg,#059669 0%,#0d9488 100%) !important;color:#fff !important;
    font-weight:bold !important;border:none !important;width:100% !important;padding:14px 0 !important;
    border-radius:10px !important;box-shadow:0 4px 18px rgba(5,150,105,.35) !important;}
div[data-testid="stDownloadButton"]>button:hover{transform:translateY(-2px);}

.badge-pos{background:linear-gradient(135deg,#7f1d1d,#991b1b);color:#fca5a5;
    padding:8px 20px;border-radius:8px;font-weight:bold;font-family:sans-serif;font-size:14px;
    display:inline-block;letter-spacing:1px;border:1px solid rgba(252,165,165,.2);}
.badge-neg{background:linear-gradient(135deg,#064e3b,#065f46);color:#6ee7b7;
    padding:8px 20px;border-radius:8px;font-weight:bold;font-family:sans-serif;font-size:14px;
    display:inline-block;letter-spacing:1px;border:1px solid rgba(110,231,183,.3);}
.type-chip{display:inline-block;padding:4px 12px;border-radius:20px;font-family:monospace;font-size:11px;
    font-weight:bold;letter-spacing:.8px;margin:3px;border:1px solid;}
.chip-bact{background:rgba(239,68,68,.12);color:#fca5a5;border-color:rgba(239,68,68,.3);}
.chip-virl{background:rgba(168,85,247,.12);color:#d8b4fe;border-color:rgba(168,85,247,.3);}
.chip-fung{background:rgba(234,179,8,.12);color:#fde68a;border-color:rgba(234,179,8,.3);}
.chip-unkn{background:rgba(148,163,184,.10);color:#94a3b8;border-color:rgba(148,163,184,.25);}
.chip-left{background:rgba(59,130,246,.12);color:#93c5fd;border-color:rgba(59,130,246,.3);}
.chip-right{background:rgba(20,184,166,.12);color:#5eead4;border-color:rgba(20,184,166,.3);}
.chip-bil{background:rgba(245,158,11,.12);color:#fcd34d;border-color:rgba(245,158,11,.3);}

.spill{background:linear-gradient(165deg,rgba(45,212,191,.09) 0%,rgba(13,148,136,.03) 100%);
    border:1px solid rgba(45,212,191,.18);border-radius:14px;padding:14px 10px;
    text-align:center;transition:all .4s cubic-bezier(.22,1,.36,1);
    box-shadow:0 6px 20px rgba(0,0,0,.25);
    min-height:74px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:4px;}
.spill:hover{transform:translateY(-4px) scale(1.02);border-color:rgba(45,212,191,.5);}
.spill .sl{font-family:monospace;font-size:10px;color:#5eead4;letter-spacing:1px;text-transform:uppercase;opacity:.85;}
.spill .sv{font-family:sans-serif;font-size:16px;font-weight:800;color:#fff;line-height:1.2;}

.prow{display:flex;align-items:center;gap:12px;margin-bottom:14px;}
.plbl{width:110px;font-family:monospace;font-size:11px;letter-spacing:1px;text-transform:uppercase;color:#94a3b8;flex-shrink:0;}
.ptrack{flex:1;height:13px;background:rgba(255,255,255,.04);border:1px solid rgba(45,212,191,.12);border-radius:999px;overflow:hidden;}
.pfill{height:100%;border-radius:999px;animation:barF 1.1s cubic-bezier(.22,1,.36,1) both;}
@keyframes barF{from{width:0%}}
.pf-pneu{background:linear-gradient(90deg,#f87171,#dc2626);box-shadow:0 0 14px rgba(248,113,113,.45);}
.pf-norm{background:linear-gradient(90deg,#2dd4bf,#059669);box-shadow:0 0 14px rgba(45,212,191,.45);}
.pval{width:52px;text-align:right;font-family:sans-serif;font-weight:800;font-size:13px;color:#fff;flex-shrink:0;}

.xray-panel-wrap{
    background:#0a1628;
    border-radius:14px;
    border:1px solid rgba(45,212,191,.20);
    overflow:hidden;
    box-shadow:0 12px 40px rgba(0,0,0,.6), inset 0 1px 0 rgba(255,255,255,.04);
}
.xray-panel-header{
    padding:10px 16px;
    background:rgba(13,148,136,.08);
    border-bottom:1px solid rgba(45,212,191,.15);
    font-family:monospace;
    font-size:11px;
    font-weight:bold;
    letter-spacing:2px;
    text-transform:uppercase;
    color:#94a3b8;
    display:flex;
    align-items:center;
    gap:8px;
}
.xray-panel-header.teal{color:#2dd4bf;}
.xray-panel-body{
    position:relative;
    width:100%;
    background:#050e1a;
}
.xray-panel-body img{
    width:100%;
    height:260px;
    object-fit:cover;
    display:block;
}
.consolidation-badge{
    position:absolute;
    bottom:12px;
    right:12px;
    background:linear-gradient(135deg,#dc2626,#991b1b);
    color:#fff;
    font-family:monospace;
    font-size:10px;
    font-weight:bold;
    letter-spacing:1.5px;
    padding:5px 12px;
    border-radius:6px;
    border:1px solid rgba(252,165,165,.4);
    box-shadow:0 4px 16px rgba(220,38,38,.5), 0 0 24px rgba(220,38,38,.3);
    text-transform:uppercase;
    animation:fadeUp .5s ease both;
}
.normal-badge{
    position:absolute;
    bottom:12px;
    right:12px;
    background:linear-gradient(135deg,#065f46,#064e3b);
    color:#6ee7b7;
    font-family:monospace;
    font-size:10px;
    font-weight:bold;
    letter-spacing:1.5px;
    padding:5px 12px;
    border-radius:6px;
    border:1px solid rgba(110,231,183,.35);
    box-shadow:0 4px 16px rgba(6,95,70,.5);
    text-transform:uppercase;
    animation:fadeUp .5s ease both;
}
.focus-label{
    position:absolute;
    top:10px;
    left:10px;
    background:rgba(0,0,0,.65);
    color:#2dd4bf;
    font-family:monospace;
    font-size:9px;
    letter-spacing:1px;
    padding:3px 8px;
    border-radius:4px;
    border:1px solid rgba(45,212,191,.3);
}

.cbubble{background:linear-gradient(165deg,rgba(45,212,191,.08) 0%,rgba(13,148,136,.03) 100%);
    backdrop-filter:blur(16px);border:1px solid rgba(45,212,191,.16);border-left:4px solid #2dd4bf;
    padding:18px;border-radius:12px;font-family:sans-serif;font-size:13px;line-height:1.7;
    color:#e2f8f0;margin-bottom:12px;animation:fadeUp .7s ease both;
    box-shadow:0 8px 24px rgba(0,0,0,.3);}
.disc{background:linear-gradient(135deg,rgba(30,27,11,.9),rgba(40,33,10,.7));
    border:1px solid #78350f;color:#fcd34d;padding:10px 16px;border-radius:10px;
    font-family:monospace;font-size:11px;letter-spacing:.5px;margin-bottom:18px;}
.shead{background:linear-gradient(165deg,rgba(45,212,191,.08) 0%,rgba(5,150,105,.03) 100%);
    backdrop-filter:blur(18px);border:1px solid rgba(45,212,191,.18);padding:22px 24px;
    border-radius:16px;margin-bottom:18px;display:flex;justify-content:space-between;align-items:center;
    animation:fadeUp .6s ease both;box-shadow:0 14px 40px rgba(0,0,0,.4),inset 0 1px 0 rgba(255,255,255,.05);
    position:relative;overflow:hidden;}
.shead::before{content:"";position:absolute;top:-60%;left:-10%;width:50%;height:220%;
    background:linear-gradient(120deg,rgba(45,212,191,0) 0%,rgba(45,212,191,.07) 50%,rgba(45,212,191,0) 100%);
    transform:rotate(15deg);pointer-events:none;}
.scap{font-size:11px;font-family:monospace;color:#5eead4;font-weight:bold;text-transform:uppercase;
    letter-spacing:1.5px;display:inline-block;padding-bottom:4px;border-bottom:2px solid rgba(45,212,191,.25);margin-bottom:4px;}
.ldr-wrap{text-align:center;padding:100px 0;min-height:60vh;display:flex;flex-direction:column;align-items:center;justify-content:center;animation:fadeIn .4s ease both;}
.ldr-ring{width:72px;height:72px;border-radius:50%;border:4px solid rgba(45,212,191,.12);border-top:4px solid #2dd4bf;border-right:4px solid #059669;animation:spin .9s linear infinite;}
.ldr-inner{position:absolute;top:12px;left:12px;width:48px;height:48px;border-radius:50%;border:3px solid rgba(94,234,212,.12);border-bottom:3px solid #5eead4;animation:spin 1.4s linear infinite reverse;}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes fadeUp{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:translateY(0)}}
@keyframes fadeIn{from{opacity:0}to{opacity:1}}
@keyframes glow{0%,100%{opacity:.35;transform:scale(1)}50%{opacity:1;transform:scale(1.3)}}
div[data-testid="stAlertContainer"]{
    background:linear-gradient(165deg,rgba(45,212,191,.08) 0%,rgba(13,148,136,.03) 100%) !important;
    border:1px solid rgba(45,212,191,.18) !important;border-left:4px solid #2dd4bf !important;
    border-radius:12px !important;color:#e2f8f0 !important;
    box-shadow:0 8px 24px rgba(0,0,0,.3) !important;}
div[data-testid="stAlertContainer"] p{color:#e2f8f0 !important;font-size:13px !important;line-height:1.6 !important;}
</style>
<div class="amb amb1"></div>
<div class="amb amb2"></div>
<div class="amb amb3"></div>
""", unsafe_allow_html=True)

# ============================================================
# SVG BRAND MARK
# ============================================================
def brand_svg(size=40):
    return f"""<div style="width:{size}px;height:{size}px;flex-shrink:0;">
<svg viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
<defs><linearGradient id="mg{size}" x1="0%" y1="0%" x2="100%" y2="100%">
<stop offset="0%" stop-color="#2dd4bf"/><stop offset="100%" stop-color="#059669"/>
</linearGradient></defs>
<circle cx="32" cy="32" r="28" fill="url(#mg{size})" opacity=".16"/>
<circle cx="32" cy="32" r="22" fill="none" stroke="url(#mg{size})" stroke-width="2" opacity=".5"/>
<path d="M32 14 C 24 14 18 22 18 32 C 18 42 22 48 27 48 C 30 48 30 42 30 38 L 30 18"
      fill="none" stroke="#6ee7b7" stroke-width="2.2" stroke-linecap="round" opacity=".85"/>
<path d="M32 14 C 40 14 46 22 46 32 C 46 42 42 48 37 48 C 34 48 34 42 34 38 L 34 18"
      fill="none" stroke="#6ee7b7" stroke-width="2.2" stroke-linecap="round" opacity=".85"/>
<rect x="29" y="22" width="6" height="20" rx="2" fill="#fff"/>
<rect x="22" y="29" width="20" height="6" rx="2" fill="#fff"/>
</svg></div>"""

# ============================================================
# SESSION STATE
# ============================================================
for k, v in [("chat_history", []), ("diagnostic_context", None), ("pipeline_active", False)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# MODEL LOADING
# ============================================================
# ============================================================
# MODEL LOADING
# ============================================================
@st.cache_resource
def load_model_cached():
    paths = []
    ep = os.getenv("MODEL_PATH")
    if ep: paths.append(ep)
    paths += [
        os.path.join("storage", "models", "xray_model_best.onnx"),
        os.path.join("models", "xray_model_best.onnx"),
        "xray_model_best.onnx",
    ]
    for p in paths:
        if os.path.exists(p):
            # Forcing it to load using your updated ONNX classifier class
            from core.classifier import XRayClassifier
            return XRayClassifier(p)
    return None
# ============================================================
# GRAD-CAM
# ============================================================
def _last_conv(m):
    for layer in reversed(m.layers):
        if hasattr(layer,"layers") and layer.layers:
            r,rm = _last_conv(layer)
            if r: return r,rm
        try:
            s = layer.output_shape
        except: continue
        if isinstance(s,list): s=s[0]
        if s and len(s)==4: return layer,m
    return None,None

def gradcam(img_array, model):
    cl, om = _last_conv(model)
    if cl is None:
        try: h,w=int(model.inputs[0].shape[1]),int(model.inputs[0].shape[2])
        except: h,w=224,224
        return np.zeros((h,w),dtype=np.float32)
    gm = tf.keras.models.Model([om.inputs],[cl.output,om.output])
    with tf.GradientTape() as tape:
        inp = tf.cast(img_array,tf.float32)
        co,pred = gm(inp)
        if isinstance(pred,list): pred=pred[0]
        loss=pred[:,0]
    grads=tape.gradient(loss,co)
    gg=tf.reduce_mean(grads,axis=(0,1,2))
    co=co[0]
    hm=co@gg[...,tf.newaxis]
    hm=tf.squeeze(hm)
    hm=tf.maximum(hm,0)/(tf.reduce_max(hm)+1e-8)
    return hm.numpy()

# ============================================================
# RED-HOT LUT
# ============================================================
def red_hot_lut():
    lut=np.zeros((256,1,3),dtype=np.uint8)
    for i in range(256):
        t=i/255.0
        if t<0.5:   r=int(255*(t/0.5)); g=0; b=0
        elif t<0.85:r=255; g=int(165*((t-0.5)/0.35)); b=0
        else:       r=255; g=int(165+(255-165)*((t-0.85)/0.15)); b=int(255*((t-0.85)/0.15))
        lut[i,0]=[b,g,r]
    return lut

_LUT = red_hot_lut()

# ============================================================
# BUILD OVERLAY
# ============================================================
def build_overlay(raw, hm_r, is_pos=True):
    if not is_pos:
        return np.clip(raw.astype(np.float32), 0, 255).astype(np.uint8)
    hm = np.nan_to_num(hm_r, nan=0.0, posinf=1.0, neginf=0.0)
    hm = np.clip(hm, 0, 1)
    hm_max = float(hm.max())
    if hm_max > 1e-6:
        hm = hm / hm_max
    else:
        h, w = hm.shape
        cy, cx = h // 2, w // 2
        ys, xs = np.mgrid[0:h, 0:w]
        hm = np.exp(-((ys - cy) ** 2 / (0.2 * h) ** 2 + (xs - cx) ** 2 / (0.2 * w) ** 2))
        hm = hm / hm.max()
    hm_s = cv2.GaussianBlur(hm.astype(np.float32), (0, 0), sigmaX=28, sigmaY=28)
    hm_s = np.clip(hm_s / max(float(hm_s.max()), 1e-6), 0, 1)
    thr = max(float(np.percentile(hm_s, 45)), 0.10)
    colored = cv2.applyColorMap(np.uint8(255 * hm_s), _LUT)
    alpha_base = np.clip((hm_s - thr) / max(1.0 - thr, 1e-6), 0, 1)
    thr_peak = max(float(np.percentile(hm_s, 80)), 0.50)
    alpha_peak = np.clip((hm_s - thr_peak) / max(1.0 - thr_peak, 1e-6), 0, 1)
    alpha = np.clip(alpha_base * 0.80 + alpha_peak * 0.20, 0, 1)[..., np.newaxis]
    dark_raw = raw.astype(np.float32) * 0.50
    blended = dark_raw * (1 - alpha) + colored.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)

# ============================================================
# LOCATION / FOCUS
# ============================================================
def focus_region_label(hm):
    h,w=hm.shape; total=hm.sum()
    if total<=0: return "Diffuse / No Dominant Region"
    ys,xs=np.mgrid[0:h,0:w]
    cy=float((ys*hm).sum()/total)/h
    cx=float((xs*hm).sum()/total)/w
    v="Upper" if cy<0.4 else ("Lower" if cy>0.6 else "Mid")
    hz="Left" if cx<0.4 else ("Right" if cx>0.6 else "Central")
    if hz=="Central" and v=="Mid": return "Central / Diffuse"
    return f"{hz} {v} Field"

def focus_box(hm):
    h,w=hm.shape; total=float(hm.sum())
    if total<=1e-6: return (0,0,1,1),(0.5,0.5),False
    ys,xs=np.mgrid[0:h,0:w]
    cy=float((ys*hm).sum()/total)/h; cx=float((xs*hm).sum()/total)/w
    vy=float((((ys/h)-cy)**2*hm).sum()/total); vx=float((((xs/w)-cx)**2*hm).sum()/total)
    sy=max(np.sqrt(vy),0.06); sx=max(np.sqrt(vx),0.06)
    x0=max(0,cx-1.5*sx); x1=min(1,cx+1.5*sx); y0=max(0,cy-1.5*sy); y1=min(1,cy+1.5*sy)
    return (x0,y0,x1,y1),(cx,cy),True

def draw_marker(img, box, cent, has):
    out=img.copy(); H,W=out.shape[:2]
    x0,y0,x1,y1=box; cx,cy=cent
    px0,py0,px1,py1=int(x0*W),int(y0*H),int(x1*W),int(y1*H)
    pcx,pcy=int(cx*W),int(cy*H)
    col=(255,255,255); thick=2
    if has:
        cv2.rectangle(out,(px0,py0),(px1,py1),col,thick,cv2.LINE_AA)
        cl=max(10,int(0.035*min(W,H)))
        cv2.line(out,(pcx-cl,pcy),(pcx+cl,pcy),col,thick,cv2.LINE_AA)
        cv2.line(out,(pcx,pcy-cl),(pcx,pcy+cl),col,thick,cv2.LINE_AA)
        cv2.circle(out,(pcx,pcy),max(4,cl//3),col,thick,cv2.LINE_AA)
    return out

def location_map_fig(box,cent,has,label):
    fig,ax=plt.subplots(figsize=(2.6,2.6),dpi=150)
    fig.patch.set_facecolor("#07140f"); ax.set_facecolor("#07140f")
    for i in range(1,3):
        ax.axvline(i/3,color="#1e3a2f",lw=1); ax.axhline(i/3,color="#1e3a2f",lw=1)
    for sp in ax.spines.values(): sp.set_color("#1e3a2f")
    x0,y0,x1,y1=box
    rect=patches.Rectangle((x0,y0),(x1-x0),(y1-y0),lw=2,
        edgecolor="#2dd4bf" if has else "#475569",
        facecolor="#2dd4bf" if has else "#475569",alpha=.18 if has else .10)
    ax.add_patch(rect)
    cx,cy=cent
    if has:
        ax.plot(cx,cy,marker="+",markersize=14,color="#f87171",mew=2.5)
        ax.plot(cx,cy,marker="o",markersize=6,color="#f87171",fillstyle="none",mew=1.5)
    else:
        ax.plot(cx,cy,marker="o",markersize=5,color="#94a3b8")
    ax.set_xlim(0,1); ax.set_ylim(1,0); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(label,color="#5eead4",fontsize=8,fontfamily="monospace",pad=7)
    plt.tight_layout(); return fig

# ============================================================
# ANIMATED DONUT
# ============================================================
def animated_donut(score, is_pos, height=210, key="dnt"):
    # FIX: show confidence in the ACTUAL result, not always pneumonia prob
    confidence = score if is_pos else (1.0 - score)
    pct = round(max(0.0, min(100.0, confidence * 100)), 1)
    fill = "#ef4444" if is_pos else "#2dd4bf"
    status = "POSITIVE" if is_pos else "NEGATIVE"
    html = f"""
<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
            height:{height}px;background:transparent;font-family:sans-serif;">
  <canvas id="{key}" width="170" height="170" style="max-width:100%;"></canvas>
  <div id="{key}_lbl" style="font-size:13px;font-weight:800;color:#fff;margin-top:8px;letter-spacing:.5px;">
    {status} &mdash; 0.0%
  </div>
</div>
<script>
(function(){{
  var canvas=document.getElementById("{key}");
  var ctx=canvas.getContext("2d");
  var W=170,R=62,cx=W/2,cy=W/2;
  var target={pct},fill="{fill}",label="{status}";
  var lblEl=document.getElementById("{key}_lbl");
  var cur=0;
  function draw(v){{
    ctx.clearRect(0,0,W,W);
    ctx.beginPath();ctx.arc(cx,cy,R,0,Math.PI*2);
    ctx.strokeStyle="#1e3a2f";ctx.lineWidth=22;ctx.stroke();
    if(v>0){{
      ctx.beginPath();ctx.arc(cx,cy,R,-Math.PI/2,-Math.PI/2+(v/100)*Math.PI*2);
      ctx.strokeStyle=fill;ctx.lineWidth=22;ctx.lineCap="round";ctx.stroke();
    }}
    ctx.fillStyle="#ffffff";ctx.font="bold 22px sans-serif";
    ctx.textAlign="center";ctx.textBaseline="middle";
    ctx.fillText(v.toFixed(1)+"%",cx,cy-6);
    ctx.fillStyle="#5eead4";ctx.font="10px monospace";
    ctx.fillText("CONFIDENCE",cx,cy+16);
    if(lblEl)lblEl.textContent=label+" \u2014 "+v.toFixed(1)+"%";
  }}
  draw(0);
  var step=Math.max(target/40,0.5);
  var timer=setInterval(function(){{
    cur+=step;
    if(cur>=target){{cur=target;draw(cur);clearInterval(timer);}}
    else draw(cur);
  }},16);
}})();
</script>"""
    return html

# ============================================================
# CHIP HELPERS
# ============================================================
_TYPE_CHIP={"Bacterial":"chip-bact","Viral":"chip-virl","Fungal":"chip-fung","Unknown":"chip-unkn"}
_SIDE_CHIP={"Left Lung":"chip-left","Right Lung":"chip-right","Bilateral":"chip-bil","Unknown":"chip-unkn"}
def type_chip(t): return f'<span class="type-chip {_TYPE_CHIP.get(t,"chip-unkn")}">&#9877; {t}</span>'
def side_chip(s):
    ic={"Left Lung":"&#9664;","Right Lung":"&#9654;","Bilateral":"&#9664;&#9654;","Unknown":"?"}
    return f'<span class="type-chip {_SIDE_CHIP.get(s,"chip-unkn")}">{ic.get(s,"")} {s}</span>'

# ============================================================
# GROQ AI
# ============================================================
_GROQ_MODEL = "llama-3.3-70b-versatile"

def _groq_call(prompt: str) -> str:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return "AI features not enabled — set GROQ_API_KEY in your .env file."
    url = "https://api.groq.com/openai/v1/chat/completions"
    payload = {
        "model": _GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "temperature": 0.4,
    }
    try:
        resp = _requests.post(
            url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json=payload,
            timeout=45,
        )
        if not resp.ok:
            return f"Groq API error {resp.status_code}: {resp.text[:400]}"
        return resp.json()["choices"][0]["message"]["content"]
    except _requests.exceptions.Timeout:
        return "Groq request timed out — try again."
    except Exception as e:
        return f"Groq call failed: {e}"


def groq_summary(patient, prob, focus, ptype, side, is_pos):
    label = "positive (pneumonia suspected)" if is_pos else "negative (no pneumonia detected)"
    confidence = prob if is_pos else (1 - prob)
    if is_pos:
        focus_instruction = f"Grad-CAM focus: {focus}. Briefly explain what this focus region indicates in section 2."
    else:
        focus_instruction = ("No Grad-CAM focus region is shown for negative results. "
                              "In section 2, explain that since the screen was negative, "
                              "no specific attention region is highlighted.")
    prompt = f"""Summarise output of an automated chest X-ray screening model for a research dashboard.
RULES: No clinical findings, no lobe names, no medication. Not a diagnosis.
Patient: {patient['age']}-yr {patient['gender']}, location: {patient.get('location', 'N/A')}.
Type (user-entered): {ptype}. Side (user-entered): {side}.
Model: {label}, confidence {confidence:.1%}. {focus_instruction}
Write 3 short plain-language sections:
1. What model output means (screening aid, not diagnosis).
2. Explain the Grad-CAM focus / attention information as instructed above.
3. Next step: consult licensed physician."""
    return _groq_call(prompt)


def groq_chat(query, history, ctx=None):
    hist = "\n".join(f"{r.upper()}: {m}" for r, m in history[-10:])
    ctx_b = "" if not ctx else (
        f"\nSession: classifier={ctx.get('label')}, confidence={ctx.get('confidence')}, "
        f"focus={ctx.get('focus')}, type={ctx.get('ptype')}, side={ctx.get('side')}"
    )
    prompt = f"""You are an assistant in PNEUMONIA.AI (research X-ray screening tool).
Explain how the tool works. NEVER diagnose or recommend medication. 2-5 sentences max.{ctx_b}
Conversation:\n{hist}\nUSER: {query}\nASSISTANT:"""
    return _groq_call(prompt)

# ============================================================
# PDF REPORT
# ============================================================
def make_pdf(patient, prob, focus, ptype, side, raw_img, overlay_img, loc_fig, is_pos):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm,
                             topMargin=14*mm, bottomMargin=14*mm)
    W = A4[0] - 36*mm

    teal  = colors.HexColor("#0d9488")
    red   = colors.HexColor("#b91c1c")
    green = colors.HexColor("#047857")
    dark  = colors.HexColor("#0f172a")
    muted = colors.HexColor("#64748b")
    light = colors.HexColor("#f0fdfa")
    grid  = colors.HexColor("#ccfbf1")
    result_col = red if is_pos else green
    # FIX: confidence in the actual result
    confidence = prob if is_pos else (1 - prob)

    def S(n, **kw):
        d = dict(fontName="Helvetica", fontSize=10, leading=14, textColor=dark)
        d.update(kw)
        return ParagraphStyle(n, **d)

    TITLE   = S("TITLE", fontName="Helvetica-Bold", fontSize=20, leading=24, textColor=teal)
    SUBT    = S("SUBT", fontSize=9, textColor=muted, spaceAfter=2)
    REPID   = S("REPID", fontSize=8, textColor=muted, alignment=TA_LEFT)
    SH      = S("SH", fontName="Helvetica-Bold", fontSize=12, leading=14, textColor=dark, spaceBefore=10, spaceAfter=6)
    ML      = S("ML", fontSize=8, textColor=muted)
    BD      = S("BD", fontName="Helvetica-Bold")
    RESULT  = S("RESULT", fontName="Helvetica-Bold", fontSize=16, leading=20, textColor=result_col)
    CONF    = S("CONF", fontName="Helvetica-Bold", fontSize=14, leading=18, textColor=result_col)
    TL      = S("TL", fontName="Helvetica-Bold", textColor=teal)
    BODY    = S("BODY", fontSize=9.5, leading=14, textColor=dark)
    DC      = S("DC", fontName="Helvetica-Oblique", fontSize=8, leading=12, textColor=muted)
    FT      = S("FT", fontSize=7, leading=10, textColor=colors.HexColor("#94a3b8"), alignment=TA_CENTER)
    IL      = S("IL", fontSize=7, fontName="Helvetica-Bold", textColor=muted, alignment=TA_CENTER, spaceAfter=3)

    story = []

    report_no = datetime.now().strftime("PNA-%Y%m%d-%H%M%S")
    header_tbl = Table(
        [
    [Paragraph("<b>INVESTIGATIONAL RADIOGRAPHIC EVALUATION REPORT</b>", TITLE)],
    [Paragraph("<i>Clinical Research &amp; Academic Methodology Protocol — Non-Diagnostic Record</i>", SUBT)],
    [Paragraph(f"Accession No: {report_no} | Processed: {datetime.now().strftime('%d-%b-%Y | %H:%M:%S')} (IST)", REPID)]
],
colWidths=[W])
    header_tbl.setStyle(TableStyle([
        ("LEFTPADDING", (0,0), (-1,-1), 0),
        ("TOPPADDING", (0,0), (-1,-1), 1),
        ("BOTTOMPADDING", (0,0), (-1,-1), 1),
    ]))
    story.append(header_tbl)
    story.append(HRFlowable(width="100%", thickness=3, color=teal, spaceBefore=6, spaceAfter=10))

    story.append(Paragraph("1. Patient Information", SH))
    patient_tbl = Table([
        [Paragraph("PATIENT NAME", ML), Paragraph("AGE / GENDER", ML), Paragraph("LOCATION", ML)],
        [Paragraph(patient["name"], BD),
         Paragraph(f"{patient['age']} yrs / {patient['gender']}", BD),
         Paragraph(patient.get("location", "N/A"), BD)],
        [Paragraph("EXAMINATION DATE", ML), Paragraph("PNEUMONIA TYPE (USER-ENTERED)", ML), Paragraph("AFFECTED SIDE (USER-ENTERED)", ML)],
        [Paragraph(datetime.now().strftime("%d %B %Y"), BD),
         Paragraph(ptype, BD),
         Paragraph(side, BD)],
    ], colWidths=[W/3.0]*3)
    patient_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), light),
        ("BOX", (0,0), (-1,-1), .75, grid),
        ("INNERGRID", (0,0), (-1,-1), .5, grid),
        ("TOPPADDING", (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ("LEFTPADDING", (0,0), (-1,-1), 10),
    ]))
    story.append(patient_tbl)
    story.append(Spacer(1, 10))

    story.append(Paragraph("2. Screening Result", SH))
    result_label = "PNEUMONIA SUSPECTED" if is_pos else "NO PNEUMONIA DETECTED"
    focus_text = focus if is_pos else "Not applicable (negative screen - no attention region highlighted)"
    result_tbl = Table([
        [Paragraph("CLASSIFICATION", ML), Paragraph("MODEL CONFIDENCE", ML), Paragraph("GRAD-CAM FOCUS REGION", ML)],
        [Paragraph(result_label, RESULT),
         Paragraph(f"{confidence*100:.1f}%", CONF),
         Paragraph(focus_text, TL if is_pos else BODY)],
    ], colWidths=[W*0.32, W*0.22, W*0.46])
    result_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e0fdf4")),
        ("BACKGROUND", (0,1), (-1,1), colors.white),
        ("BOX", (0,0), (-1,-1), .75, grid),
        ("INNERGRID", (0,0), (-1,-1), .5, grid),
        ("TOPPADDING", (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
        ("LEFTPADDING", (0,0), (-1,-1), 10),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(result_tbl)
    story.append(Spacer(1, 10))

    prob_tbl = Table([
        [Paragraph("CLASS", ML), Paragraph("PROBABILITY", ML)],
        [Paragraph("Pneumonia", BODY), Paragraph(f"{prob*100:.1f}%", S("pp", fontName="Helvetica-Bold", textColor=red if is_pos else dark))],
        [Paragraph("Normal", BODY), Paragraph(f"{(1-prob)*100:.1f}%", S("pn", fontName="Helvetica-Bold", textColor=green if not is_pos else dark))],
    ], colWidths=[W*0.5, W*0.5])
    prob_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), light),
        ("BOX", (0,0), (-1,-1), .75, grid),
        ("INNERGRID", (0,0), (-1,-1), .5, grid),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING", (0,0), (-1,-1), 10),
    ]))
    story.append(prob_tbl)
    story.append(Spacer(1, 10))

    rp, hp, lp, dp = "t_raw.png", "t_hm.png", "t_loc.png", "t_dnt.png"
    cv2.imwrite(rp, raw_img)

    fig_d, ax_d = plt.subplots(figsize=(2.4, 2.4), dpi=200)
    fig_d.patch.set_facecolor("white")
    donut_col = "#dc2626" if is_pos else "#047857"
    ax_d.pie([confidence, 1-confidence], colors=[donut_col, "#e2e8f0"],
             startangle=90, counterclock=False, wedgeprops=dict(width=0.4, edgecolor="none"))
    ax_d.text(0, 0.06, f"{confidence*100:.1f}%", ha="center", va="center", fontsize=12, fontweight="bold", color="#0f172a")
    ax_d.text(0, -0.16, "CONFIDENCE", ha="center", va="center", fontsize=7, color="#64748b", fontfamily="monospace")
    plt.tight_layout()
    plt.savefig(dp, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig_d)

    def ic(p, lbl, w, h):
        return Table([[Paragraph(lbl, IL)], [RLImage(p, width=w, height=h)]], colWidths=[w])

    story.append(Paragraph("3. Radiograph &amp; Visualisation", SH))

    if is_pos:
        cv2.imwrite(hp, overlay_img)
        loc_fig.savefig(lp, dpi=200, facecolor=loc_fig.get_facecolor(), bbox_inches="tight")
        IW = (W - 6*mm) / 3
        IH = IW * 0.9
        cells = [
            ic(rp, "INPUT RADIOGRAPH", IW, IH),
            ic(hp, "GRAD-CAM OVERLAY + MARKER", IW, IH),
            ic(lp, "LOCATION MAP", IW, IH),
        ]
        for t in cells:
            t.setStyle(TableStyle([("ALIGN", (0,0), (-1,-1), "CENTER")]))
        it = Table([cells], colWidths=[IW]*3, rowHeights=[IH+20])
        img_files = [rp, hp, lp, dp]
    else:
        IW = (W - 4*mm) / 2
        IH = IW * 0.9
        cells = [ic(rp, "INPUT RADIOGRAPH", IW, IH)]
        for t in cells:
            t.setStyle(TableStyle([("ALIGN", (0,0), (-1,-1), "CENTER")]))
        it = Table([cells], colWidths=[IW], rowHeights=[IH+20])
        img_files = [rp, dp]

    it.setStyle(TableStyle([
        ("BOX", (0,0), (-1,-1), .75, grid),
        ("INNERGRID", (0,0), (-1,-1), .5, grid),
        ("TOPPADDING", (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
    ]))
    story.append(it)
    story.append(Spacer(1, 10))

    story.append(Paragraph("4. Impression (Non-Diagnostic)", SH))
    if is_pos:
        impression_txt = (
            f"The automated classifier flagged this radiograph as <b>pneumonia suspected</b> with "
            f"{confidence*100:.1f}% model confidence. The Grad-CAM visualisation indicates the model's attention "
            f"was concentrated in the <b>{focus}</b> region. This is a model-attention map, not a confirmed "
            f"clinical finding, and the pneumonia type/side shown above were entered by the user, not derived "
            f"by the model."
        )
    else:
        impression_txt = (
            f"The automated classifier did <b>not detect features consistent with pneumonia</b> in this "
            f"radiograph, with {confidence*100:.1f}% model confidence in the normal class. No attention "
            f"region is highlighted for negative screens. This result does not rule out other respiratory "
            f"or medical conditions."
        )
    story.append(Paragraph(impression_txt, BODY))
    story.append(Spacer(1, 10))

    story.append(Paragraph("5. Disclaimer", SH))
    disc_txt = (
       "<b>Clinical Notice:</b> This report is generated automatically for educational evaluation "
    "and clinical research methodologies only. It does not constitute a formal diagnostic finding, "
    "a certified radiological interpretation, or a substitute for an in-person medical assessment "
    "by a licensed physician.\n\n"
    "Visual overlays (such as Grad-CAM) represent computational regions of focus to highlight "
    "areas of interest on the radiograph; they do not indicate confirmed pathology or definitive "
    "tissue lesions. Clinical parameters—including suspected pneumonia type and anatomical "
    "location (affected side)—are user-entered history and are not independently extracted from the image. "
    "All findings require mandatory correlation with clinical symptoms and must be verified by a "
    "licensed healthcare provider."
    )
    disc_t = Table([[Paragraph(disc_txt, DC)]], colWidths=[W])
    disc_t.setStyle(TableStyle([
        ("BOX", (0,0), (-1,-1), .75, grid),
        ("BACKGROUND", (0,0), (-1,-1), light),
        ("TOPPADDING", (0,0), (-1,-1), 12),
        ("BOTTOMPADDING", (0,0), (-1,-1), 12),
        ("LEFTPADDING", (0,0), (-1,-1), 14),
        ("RIGHTPADDING", (0,0), (-1,-1), 14),
    ]))
    story.append(disc_t)
    story.append(Spacer(1, 14))
    story.append(HRFlowable(width="100%", thickness=1, color=grid))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        f"PNEUMONIA.AI v4.1 | Report ID: {report_no} | "
        f"Generated {datetime.now().strftime('%d %b %Y %H:%M')} | "
        f"Automated research tool - NOT for clinical use.", FT))

    doc.build(story)
    for f in img_files:
        if os.path.exists(f):
            os.remove(f)
    buf.seek(0)
    return buf

# ============================================================
# LOAD MODEL
# ============================================================
model = load_model_cached()

# ============================================================
# TOP BAR
# ============================================================
st.markdown(f"""
<div class="topbar">
  <div style="display:flex;align-items:center;gap:12px;">
    {brand_svg(36)}
    <div style="font-family:sans-serif;font-weight:800;color:#fff;letter-spacing:1px;font-size:18px;">
      PNEUMONIA<span style="color:#2dd4bf;">.AI</span>
      <span style="font-size:10px;background:linear-gradient(135deg,#0f2a20,#134e3a);
        padding:3px 8px;border-radius:6px;margin-left:6px;color:#6ee7b7;
        border:1px solid rgba(45,212,191,.2);">V4.1</span>
    </div>
  </div>
  <div style="font-family:monospace;font-size:11px;color:#94a3b8;display:flex;align-items:center;gap:6px;">
    PIPELINE:
    <span style="color:#2dd4bf;font-weight:bold;display:flex;align-items:center;gap:5px;">
      <span style="width:8px;height:8px;border-radius:50%;background:#2dd4bf;box-shadow:0 0 8px #2dd4bf;"></span>
      ONLINE
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# INTAKE PORTAL
# ============================================================
if not st.session_state.pipeline_active:
    ph=st.empty()
    with ph.container():
        _,cc,_=st.columns([1,2.4,1])
        with cc:
            st.markdown('<div class="pform">', unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown(f"""
<div style="display:flex;flex-direction:column;align-items:center;margin-bottom:4px;">
  <div style="position:relative;width:72px;height:72px;margin:0 auto 12px;">
    <div style="position:absolute;inset:-12px;border-radius:50%;
      background:radial-gradient(circle,rgba(45,212,191,.35) 0%,transparent 70%);
      animation:glow 2.8s ease-in-out infinite;"></div>
    <div style="position:relative;z-index:1;width:100%;height:100%;">{brand_svg(72)}</div>
  </div>
  <div style="font-family:sans-serif;font-weight:800;font-size:24px;letter-spacing:1.5px;
    background:linear-gradient(135deg,#fff 30%,#6ee7b7 100%);
    -webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;
    text-align:center;">PNEUMONIA.AI INTAKE PORTAL</div>
  <div style="font-family:monospace;font-size:11px;color:#5eead4;letter-spacing:1.5px;
    text-transform:uppercase;opacity:.75;margin-bottom:24px;text-align:center;">
    Research Demo &mdash; Complete all fields before running pipeline</div>
</div>""", unsafe_allow_html=True)

                st.markdown("""<div style="text-align: center; color: #4ECDC4; font-size: 1.5rem; font-weight: bold; padding: 10px;">🫁 Welcome to Pneumonia AI</div>""", unsafe_allow_html=True)

                st.markdown('<div class="flabel-g">Patient Name</div>', unsafe_allow_html=True)
                p_name=st.text_input("name_",value="Patient Name",label_visibility="collapsed")
                c1,c2=st.columns(2)
                with c1:
                    st.markdown('<div class="flabel-g">Age</div>', unsafe_allow_html=True)
                    p_age=st.number_input("age_",min_value=0,max_value=120,value=45,label_visibility="collapsed")
                with c2:
                    st.markdown('<div class="flabel-g">Gender</div>', unsafe_allow_html=True)
                    p_gender=st.selectbox("gender_",["Male","Female","Other"],label_visibility="collapsed")
                st.markdown('<div class="flabel-g">Patient Location</div>', unsafe_allow_html=True)
                p_location=st.text_input("loc_",value="",placeholder="e.g. Balasore, Odisha, IN",label_visibility="collapsed")
                c3,c4=st.columns(2)
                with c3:
                    st.markdown('<div class="flabel-t">Pneumonia Type</div>', unsafe_allow_html=True)
                    p_type=st.selectbox("ptype_",["Unknown","Bacterial","Viral","Fungal"],label_visibility="collapsed")
                with c4:
                    st.markdown('<div class="flabel-t">Affected Side</div>', unsafe_allow_html=True)
                    p_side=st.selectbox("pside_",["Unknown","Left Lung","Right Lung","Bilateral"],label_visibility="collapsed")
                st.markdown('<div class="flabel-t">Chest X-Ray Image</div>', unsafe_allow_html=True)
                st.markdown(f'<div style="display:flex;justify-content:center;margin-bottom:8px;opacity:.8;">{brand_svg(28)}</div>',unsafe_allow_html=True)
                uploaded=st.file_uploader("xray_",type=["png","jpg","jpeg"],label_visibility="collapsed")

                if uploaded:
                    raw_bytes=uploaded.getvalue()
                    arr=np.asarray(bytearray(raw_bytes),dtype=np.uint8)
                    prev=cv2.imdecode(arr,cv2.IMREAD_COLOR)
                    if prev is None:
                        st.error("Could not decode image.")
                    else:
                        st.write("")
                        st.markdown('<span class="scap">Preview &mdash; Ready For Pipeline</span>',unsafe_allow_html=True)
                        st.write("")
                        pc1,pc2=st.columns([1,1.2],gap="medium")
                        with pc1:
                            st.image(cv2.cvtColor(prev,cv2.COLOR_BGR2RGB),use_container_width=True)
                            st.markdown(f'<div style="text-align:center;font-family:monospace;font-size:10px;color:#5eead4;margin-top:4px;">{uploaded.name}</div>',unsafe_allow_html=True)
                        with pc2:
                            components.html(animated_donut(0,False,height=170,key="dnt_preview"),height=180)
                            st.markdown('<div style="text-align:center;font-family:monospace;font-size:10px;color:#94a3b8;letter-spacing:1px;text-transform:uppercase;margin-top:4px;">Score shown after pipeline runs</div>',unsafe_allow_html=True)
                        st.markdown(f'<div style="margin-top:10px;display:flex;gap:6px;flex-wrap:wrap;">{type_chip(p_type)}{side_chip(p_side)}</div>',unsafe_allow_html=True)

                _,bc,_=st.columns([1,1,1])
                with bc:
                    go=st.button("Run Screening Pipeline",type="primary")

            if go:
                if not uploaded:
                    st.error("Please upload a chest X-ray image.")
                else:
                    st.session_state.p_name=p_name; st.session_state.p_age=p_age
                    st.session_state.p_gender=p_gender
                    st.session_state.p_location=p_location.strip() if p_location else "Not specified"
                    st.session_state.p_type=p_type; st.session_state.p_side=p_side
                    st.session_state.uploaded_bytes=uploaded.getvalue()
                    ph.empty()
                    with ph.container():
                        st.markdown("""
<div class="ldr-wrap">
  <div style="position:relative;width:72px;height:72px;margin:0 auto 20px;">
    <div style="position:absolute;inset:-18px;border-radius:50%;background:radial-gradient(circle,rgba(45,212,191,.35) 0%,transparent 70%);animation:glow 2.2s ease-in-out infinite;"></div>
    <div class="ldr-ring"></div><div class="ldr-inner"></div>
  </div>
  <div style="font-family:monospace;color:#fff;font-size:15px;letter-spacing:3px;text-transform:uppercase;font-weight:700;margin-bottom:5px;">Running Pipeline</div>
  <div style="font-family:monospace;color:#5eead4;font-size:11px;letter-spacing:1px;margin-bottom:22px;">Analysing radiograph &mdash; please wait</div>
  <div style="display:inline-flex;flex-direction:column;gap:10px;text-align:left;font-family:monospace;font-size:12px;color:#94a3b8;background:rgba(13,148,136,.05);border:1px solid rgba(45,212,191,.12);border-radius:12px;padding:16px 22px;">
    <div style="display:flex;align-items:center;gap:10px;"><span style="width:8px;height:8px;border-radius:50%;background:#2dd4bf;box-shadow:0 0 8px rgba(45,212,191,.6);animation:glow 1.4s ease-in-out infinite;"></span> Ingesting X-ray image</div>
    <div style="display:flex;align-items:center;gap:10px;"><span style="width:8px;height:8px;border-radius:50%;background:#2dd4bf;box-shadow:0 0 8px rgba(45,212,191,.6);animation:glow 1.4s .25s ease-in-out infinite;"></span> Running ResNet50 classifier</div>
    <div style="display:flex;align-items:center;gap:10px;"><span style="width:8px;height:8px;border-radius:50%;background:#2dd4bf;box-shadow:0 0 8px rgba(45,212,191,.6);animation:glow 1.4s .5s ease-in-out infinite;"></span> Computing red-hot Grad-CAM</div>
    <div style="display:flex;align-items:center;gap:10px;"><span style="width:8px;height:8px;border-radius:50%;background:#2dd4bf;box-shadow:0 0 8px rgba(45,212,191,.6);animation:glow 1.4s .75s ease-in-out infinite;"></span> Mapping focus region</div>
    <div style="display:flex;align-items:center;gap:10px;"><span style="width:8px;height:8px;border-radius:50%;background:#2dd4bf;box-shadow:0 0 8px rgba(45,212,191,.6);animation:glow 1.4s 1s ease-in-out infinite;"></span> Generating PDF report</div>
  </div>
</div>""", unsafe_allow_html=True)
                    import time; time.sleep(1.4)
                    st.session_state.pipeline_active=True
                    ph.empty(); st.rerun()
            st.markdown('</div>',unsafe_allow_html=True)

# ============================================================
# RESULTS DASHBOARD
# ============================================================
else:
        fb = np.asarray(bytearray(st.session_state.uploaded_bytes), dtype=np.uint8)
        raw_img = cv2.imdecode(fb, cv2.IMREAD_COLOR)
        resized = cv2.resize(raw_img, (224, 224))
        norm = resized.astype("float32") / 255.0
        tensor = np.expand_dims(norm, 0)

        if model is None:
            st.error("Model not found. Place model at 'storage/models/xray_model_best.onnx' and restart.")
            if st.button("Return to Portal"):
                st.session_state.pipeline_active = False
                st.rerun()
            st.stop()

        # ── INFERENCE ───────────────────────────────────────────
        result_label, final_confidence = model.predict(tensor)
        is_pos = (result_label == "PNEUMONIA")
        prob = final_confidence / 100.0 if is_pos else (1.0 - (final_confidence / 100.0))
        confidence = prob if is_pos else (1.0 - prob)

        # ── HELPER FUNCTION TO BYPASS INDENTATION TRAPS ────────
        def compute_spatial_maps(is_positive, model_obj, img_tensor, original_img):
            if not is_positive:
                heatmap_resized = np.zeros((original_img.shape[0], original_img.shape[1], 3), dtype=np.uint8)
                lbl = "Not applicable (negative screen)"
                bx, cnt, h_box = (0, 0, 1, 1), (0.5, 0.5), False
                return heatmap_resized, lbl, bx, cnt, h_box
            
            heatmap_raw = model_obj.generate_gradcam(img_tensor)
            heatmap_resized = cv2.resize(heatmap_raw, (original_img.shape[1], original_img.shape[0]))
            hm_gray = cv2.cvtColor(heatmap_resized, cv2.COLOR_BGR2GRAY) if len(heatmap_resized.shape) == 3 else heatmap_resized
            lbl = focus_region_label(hm_gray)
            bx, cnt, h_box = focus_box(hm_gray)
            return heatmap_resized, lbl, bx, cnt, h_box

        # Run the isolated function safely
        hm_r, region, box, cent, has = compute_spatial_maps(is_pos, model, tensor, raw_img)

        # ── OUTPUT GRAPHICS ROUTING ─────────────────────────────
        overlay_m = build_overlay(raw_img, hm_r, is_pos)
        if is_pos:
            overlay_m = draw_marker(overlay_m, box, cent, has)
        loc_fig = location_map_fig(box, cent, has, region)

        # ── STATE MANAGEMENT & IMAGE DATA EXTRACTION ────────────
        ptype = st.session_state.p_type
        pside = st.session_state.p_side
        plocation = st.session_state.get("p_location", "Not specified")

        def _b64(arr_bgr):
            ok, buf_enc = cv2.imencode(".png", arr_bgr)
            return base64.b64encode(buf_enc.tobytes()).decode("ascii")

        def _b64_fig(fig):
            b = io.BytesIO()
            fig.savefig(b, format="png", dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
            b.seek(0)
            return base64.b64encode(b.read()).decode("ascii")

        raw_b64 = _b64(raw_img)
        ov_b64 = _b64(overlay_m)
        loc_b64 = _b64_fig(loc_fig)

        st.markdown(f"""
<div class="shead">
  <div>
    <span style="font-size:11px;font-family:monospace;color:#5eead4;font-weight:bold;letter-spacing:1px;">ACTIVE SESSION</span>
    <h2 style="margin:4px 0 2px;font-family:sans-serif;font-weight:800;color:#fff;font-size:20px;">
      Subject: {st.session_state.p_name.upper()}</h2>
    <span style="font-family:monospace;font-size:12px;color:#2dd4bf;letter-spacing:.5px;">
      {datetime.now().strftime("%B %d, %Y")} &#8226; {st.session_state.p_age} Yrs &#8226; {st.session_state.p_gender} &#8226; {plocation}
    </span><br>
    <div style="margin-top:6px;display:flex;gap:6px;flex-wrap:wrap;">
      {type_chip(ptype)}{side_chip(pside)}
      <span class="type-chip chip-unkn">{plocation}</span>
    </div>
  </div>
  <div style="position:relative;z-index:1;">{brand_svg(44)}</div>
</div>""", unsafe_allow_html=True)

        st.markdown('<div class="disc">&#9888; Automated screening &mdash; research/education only. Not a diagnosis. Consult a physician.</div>',unsafe_allow_html=True)

        main1,main2=st.columns([5,3],gap="large")

        with main1:
            with st.container(border=True):
                badge=f'<div class="badge-pos">PNEUMONIA SUSPECTED</div>' if is_pos else f'<div class="badge-neg">NO PNEUMONIA DETECTED</div>'
                region_chip = f'<span class="type-chip chip-unkn">{region}</span>' if is_pos else ''

                st.markdown(f"""
<span class="scap">Classifier Output</span>
<div style="margin:8px 0 14px;">{badge}</div>
<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px;">
  {type_chip(ptype)}{side_chip(pside)}
  {region_chip}
</div>
<span class="scap">Model Confidence Score</span>
<h1 style="color:#2dd4bf;font-family:sans-serif;font-weight:800;margin:2px 0 20px;font-size:38px;
  text-shadow:0 0 20px rgba(45,212,191,.35);">{confidence:.1%}</h1>
""", unsafe_allow_html=True)

                pc1,pc2,pc3,pc4=st.columns(4)
                # FIX: confidence pill now shows result confidence, not raw pneumonia prob
                pills=[
                    ("Result","POSITIVE" if is_pos else "NEGATIVE","#fca5a5" if is_pos else "#6ee7b7"),
                    ("Confidence",f"{confidence:.1%}","#fff"),
                    ("Type",ptype,"#fca5a5" if ptype=="Bacterial" else "#d8b4fe" if ptype=="Viral" else "#fde68a" if ptype=="Fungal" else "#94a3b8"),
                    ("Side",pside,"#93c5fd" if "Left" in pside else "#5eead4" if "Right" in pside else "#fcd34d" if "Bilateral" in pside else "#94a3b8"),
                ]
                for col,(lbl,val,vc) in zip([pc1,pc2,pc3,pc4],pills):
                    col.markdown(f'<div class="spill"><div class="sl">{lbl}</div><div class="sv" style="color:{vc};font-size:13px;" title="{val}">{val}</div></div>',unsafe_allow_html=True)

                st.write("")
                st.markdown(f"""
<div>
<span class="scap">Class Probability Breakdown</span>
<div style="margin-top:14px;">
  <div class="prow"><div class="plbl">Pneumonia</div>
    <div class="ptrack"><div class="pfill pf-pneu" style="width:{prob*100:.1f}%;"></div></div>
    <div class="pval">{prob:.1%}</div></div>
  <div class="prow"><div class="plbl">Normal</div>
    <div class="ptrack"><div class="pfill pf-norm" style="width:{(1-prob)*100:.1f}%;"></div></div>
    <div class="pval">{(1-prob):.1%}</div></div>
</div></div>""", unsafe_allow_html=True)

                st.write("")

                # FIX: use HTML entities only — no raw unicode special chars in f-strings
                if is_pos:
                    overlay_badge = '<div class="consolidation-badge">&#9888; CONSOLIDATION ALERT</div>'
                    focus_label_html = '<div class="focus-label">&#128205; ' + region + '</div>'
                    gradcam_header_label = "GRAD-CAM SPATIAL FOCUS"
                else:
                    overlay_badge = '<div class="normal-badge">&#10003; NORMAL</div>'
                    focus_label_html = ''
                    gradcam_header_label = "RADIOGRAPH (NO FOCUS - NEGATIVE)"

                st.markdown(
                    '<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:8px;">'
                    '<div class="xray-panel-wrap">'
                    '<div class="xray-panel-header">'
                    '<svg width="12" height="12" viewBox="0 0 12 12" fill="none" style="flex-shrink:0;">'
                    '<circle cx="6" cy="6" r="5" stroke="#94a3b8" stroke-width="1.5"/>'
                    '<circle cx="6" cy="6" r="2" fill="#94a3b8"/>'
                    '</svg>'
                    'INPUT RADIOGRAPH'
                    '</div>'
                    '<div class="xray-panel-body">'
                    f'<img src="data:image/png;base64,{raw_b64}" style="width:100%;height:260px;object-fit:cover;display:block;filter:brightness(0.95) contrast(1.05);">'
                    '</div>'
                    '</div>'
                    '<div class="xray-panel-wrap">'
                    '<div class="xray-panel-header teal">'
                    '<svg width="12" height="12" viewBox="0 0 12 12" fill="none" style="flex-shrink:0;">'
                    '<circle cx="6" cy="6" r="5" stroke="#2dd4bf" stroke-width="1.5"/>'
                    '<circle cx="6" cy="6" r="2" fill="#2dd4bf"/>'
                    '</svg>'
                    f'{gradcam_header_label}'
                    '</div>'
                    '<div class="xray-panel-body">'
                    f'<img src="data:image/png;base64,{ov_b64}" style="width:100%;height:260px;object-fit:cover;display:block;">'
                    f'{focus_label_html}'
                    f'{overlay_badge}'
                    '</div>'
                    '</div>'
                    '</div>',
                    unsafe_allow_html=True
                )

                if is_pos:
                    st.markdown(
                        '<div style="margin-top:14px;">'
                        '<span class="scap">Location Map</span>'
                        '<div style="margin-top:8px;background:#07140f;border-radius:12px;border:1px solid rgba(45,212,191,.18);'
                        'padding:12px;display:flex;align-items:center;justify-content:center;height:180px;'
                        'box-shadow:0 8px 24px rgba(0,0,0,.35);">'
                        f'<img src="data:image/png;base64,{loc_b64}" style="height:160px;width:auto;max-width:100%;display:block;border-radius:8px;">'
                        '</div>'
                        '</div>',
                        unsafe_allow_html=True
                    )

            st.write("")
            with st.container(border=True):
                st.markdown('<span class="scap">Animated Confidence Score</span>',unsafe_allow_html=True)
                st.write("")
                _,dc,_=st.columns([1,2,1])
                with dc:
                    # FIX: pass confidence (not raw prob) to animated donut
                    components.html(animated_donut(prob,is_pos,height=220,key="dnt_results"),height=235)

        with main2:
            with st.container(border=True):
                st.markdown('<span class="scap">About This Result</span><br><br>',unsafe_allow_html=True)
                if is_pos:
                    focus_note = (
                        f"Grad-CAM focus: <b>{region}</b><br>"
                        "The targeted bounding box and crosshair delineate the computational Region of Interest (ROI). "
                        "This serves strictly as a visual mapping aid and does not constitute confirmed histopathology.<br><br>"
                    )
                else:
                    focus_note = (
                        "No localized abnormalities or focal opacities were segmented,   "
                        "conforming with the lack of radiographically significant patterns.<br><br>"
                    )
                st.markdown(
                    '<div class="cbubble">'
                    '<span style="color:#2dd4bf;font-weight:bold;font-family:monospace;">SCREENING SUMMARY</span><br><br>'
                    f'Classifier: <b>{"PNEUMONIA SUSPECTED" if is_pos else "NO PNEUMONIA DETECTED"}</b> &mdash; {confidence:.1%}<br><br>'
                    f'Type: {type_chip(ptype)}&nbsp; Side: {side_chip(pside)}<br>'
                    f'Location: <b>{plocation}</b><br><br>'
                    f'{focus_note}'
                    '<i>Notice: For investigational review only. Findings require formal correlation by a licensed healthcare provider.</i>'
                    '</div>',
                    unsafe_allow_html=True
                )

        # ── THREE CHARTS ROW ─────────────────────────────────
        st.write("")
        with st.container(border=True):
            st.markdown('<span class="scap">Diagnostic Charts</span>', unsafe_allow_html=True)
            st.write("")
            # FIX: charts now use `confidence` for the gauge/donut displays
            # but still show raw prob vs (1-prob) for the class breakdown bar/pie
            gauge_col = "#ef4444" if is_pos else "#2dd4bf"
            charts_html = f"""
<style>
.chart-grid{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;width:100%;}}
.chart-card{{background:rgba(10,22,40,.7);border:1px solid rgba(45,212,191,.18);border-radius:14px;
  padding:18px 10px 14px;display:flex;flex-direction:column;align-items:center;gap:8px;}}
.chart-title{{font-family:monospace;font-size:10px;color:#5eead4;letter-spacing:2px;
  text-transform:uppercase;font-weight:bold;margin-bottom:4px;}}
canvas{{display:block;}}
</style>
<div class="chart-grid">
  <div class="chart-card">
    <div class="chart-title">Bar Chart</div>
    <canvas id="barChart" width="240" height="200"></canvas>
  </div>
  <div class="chart-card">
    <div class="chart-title">Gauge Chart</div>
    <canvas id="gaugeChart" width="240" height="200"></canvas>
  </div>
  <div class="chart-card">
    <div class="chart-title">Pie Chart</div>
    <canvas id="pieChart" width="240" height="200"></canvas>
  </div>
</div>

<script>
(function(){{
  var c=document.getElementById("barChart");
  var ctx=c.getContext("2d");
  var W=240,H=200;
  var pneu={prob:.4f}, norm={1-prob:.4f};
  var vals=[pneu,norm];
  var labels=["Pneumonia","Normal"];
  var cols=["#ef4444","#2dd4bf"];
  var barW=62, gap=36, startX=18, baseY=160, maxH=120;
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle="rgba(45,212,191,.15)";ctx.fillRect(0,baseY,W,1);
  vals.forEach(function(v,i){{
    var h=v*maxH;
    var x=startX+i*(barW+gap);
    var grad=ctx.createLinearGradient(x,baseY-h,x,baseY);
    grad.addColorStop(0,cols[i]);
    grad.addColorStop(1,cols[i]+"33");
    ctx.fillStyle=grad;
    ctx.beginPath();
    if(ctx.roundRect)ctx.roundRect(x,baseY-h,barW,h,5);
    else ctx.rect(x,baseY-h,barW,h);
    ctx.fill();
    ctx.fillStyle="#ffffff";ctx.font="bold 12px sans-serif";ctx.textAlign="center";
    ctx.fillText((v*100).toFixed(1)+"%",x+barW/2,baseY-h-8);
    ctx.fillStyle="#94a3b8";ctx.font="10px monospace";ctx.textAlign="center";
    ctx.fillText(labels[i],x+barW/2,baseY+16);
  }});
}})();

(function(){{
  var c=document.getElementById("gaugeChart");
  var ctx=c.getContext("2d");
  var W=240,H=200;
  var val={confidence:.4f};
  var gcol="{gauge_col}";
  var cx=W/2, cy=130, R=80;
  var startA=Math.PI, endA=2*Math.PI;
  var fillA=startA+(val*(endA-startA));
  ctx.clearRect(0,0,W,H);
  ctx.beginPath();ctx.arc(cx,cy,R,startA,endA);
  ctx.strokeStyle="rgba(45,212,191,.15)";ctx.lineWidth=24;ctx.lineCap="round";ctx.stroke();
  var grd=ctx.createLinearGradient(cx-R,cy,cx+R,cy);
  grd.addColorStop(0,gcol+"aa");grd.addColorStop(1,gcol);
  ctx.beginPath();ctx.arc(cx,cy,R,startA,fillA);
  ctx.strokeStyle=grd;ctx.lineWidth=24;ctx.lineCap="round";ctx.stroke();
  ctx.fillStyle="#ffffff";ctx.font="bold 22px sans-serif";
  ctx.textAlign="center";ctx.textBaseline="middle";
  ctx.fillText((val*100).toFixed(1)+"%",cx,cy-12);
  ctx.fillStyle="#5eead4";ctx.font="bold 9px monospace";
  ctx.fillText("CONFIDENCE",cx,cy+10);
  ctx.fillStyle="#64748b";ctx.font="9px monospace";
  ctx.fillText("0%",cx-R+4,cy+28);ctx.fillText("100%",cx+R-4,cy+28);
}})();

(function(){{
  var c=document.getElementById("pieChart");
  var ctx=c.getContext("2d");
  var W=240,H=200;
  var pneu={prob:.4f}, norm=1-pneu;
  var pcol="{gauge_col}";
  var cx=W/2, cy=88, R=68;
  ctx.clearRect(0,0,W,H);
  var slices=[
    {{val:pneu,col:pcol}},
    {{val:norm,col:"rgba(30,58,47,.9)"}}
  ];
  var start=-Math.PI/2;
  slices.forEach(function(s){{
    var sweep=s.val*2*Math.PI;
    ctx.beginPath();ctx.moveTo(cx,cy);
    ctx.arc(cx,cy,R,start,start+sweep);
    ctx.closePath();ctx.fillStyle=s.col;ctx.fill();
    ctx.strokeStyle="#03100c";ctx.lineWidth=2;ctx.stroke();
    var mid=start+sweep/2;
    if(s.val>0.06){{
      ctx.fillStyle="#fff";ctx.font="bold 11px sans-serif";
      ctx.textAlign="center";ctx.textBaseline="middle";
      ctx.fillText((s.val*100).toFixed(1)+"%",cx+Math.cos(mid)*R*.6,cy+Math.sin(mid)*R*.6);
    }}
    start+=sweep;
  }});
  var legY=170;
  [{{col:pcol,lbl:"Pneumonia"}},{{col:"rgba(30,58,47,.9)",lbl:"Normal"}}].forEach(function(l,i){{
    var lx=i===0?28:122;
    ctx.fillStyle=l.col;ctx.strokeStyle="#5eead4";ctx.lineWidth=1;
    ctx.fillRect(lx,legY,12,12);ctx.strokeRect(lx,legY,12,12);
    ctx.fillStyle="#94a3b8";ctx.font="10px monospace";
    ctx.textAlign="left";ctx.textBaseline="middle";
    ctx.fillText(l.lbl,lx+16,legY+6);
  }});
}})();
</script>"""
            components.html(charts_html, height=280)

        # ── MEDICINE DISCLAIMER ───────────────────────────────
        st.write("")
        with st.container(border=True):
            st.markdown('<span class="scap">Medicine &amp; Treatment Information</span>', unsafe_allow_html=True)
            st.write("")
            st.markdown(
                '<div style="background:linear-gradient(135deg,rgba(239,68,68,.08),rgba(127,29,29,.12));'
                'border:1px solid rgba(239,68,68,.25);border-left:4px solid #ef4444;'
                'border-radius:12px;padding:20px 22px;font-family:sans-serif;font-size:13px;line-height:1.8;color:#e2f8f0;">'
                '<div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;">'
                '<span style="font-size:22px;">&#9877;</span>'
                '<span style="font-family:monospace;font-size:12px;font-weight:bold;color:#fca5a5;letter-spacing:1.5px;text-transform:uppercase;">'
                'Important Medical Disclaimer'
                '</span>'
                '</div>'
                '<p style="margin:0 0 12px;color:#fcd34d;font-weight:600;">'
                '&#9888; This tool does NOT provide medicine recommendations, prescriptions, or treatment advice.'
                '</p>'
                '<p style="margin:0 0 12px;">'
                'Pneumonia treatment varies significantly based on the type (bacterial, viral, fungal), '
                'severity, patient age, and underlying health conditions. Only a licensed medical professional '
                'can evaluate your condition and prescribe appropriate treatment.'
                '</p>'
                '<p style="margin:0 0 16px;">'
                'Please consult a qualified physician or visit a hospital immediately if you suspect pneumonia.'
                '</p>'
                '<div style="display:flex;gap:12px;flex-wrap:wrap;">'
                '<a href="https://www.who.int/news-room/fact-sheets/detail/pneumonia" target="_blank"'
                ' style="display:inline-flex;align-items:center;gap:6px;background:rgba(45,212,191,.12);'
                'color:#2dd4bf;border:1px solid rgba(45,212,191,.3);border-radius:8px;'
                'padding:8px 16px;font-family:monospace;font-size:11px;font-weight:bold;'
                'text-decoration:none;letter-spacing:.5px;">WHO &mdash; Pneumonia Info</a>'
                '<a href="https://www.cdc.gov/pneumonia" target="_blank"'
                ' style="display:inline-flex;align-items:center;gap:6px;background:rgba(45,212,191,.12);'
                'color:#2dd4bf;border:1px solid rgba(45,212,191,.3);border-radius:8px;'
                'padding:8px 16px;font-family:monospace;font-size:11px;font-weight:bold;'
                'text-decoration:none;letter-spacing:.5px;">CDC &mdash; Pneumonia Guide</a>'
                '<a href="https://www.nhp.gov.in" target="_blank"'
                ' style="display:inline-flex;align-items:center;gap:6px;background:rgba(45,212,191,.12);'
                'color:#2dd4bf;border:1px solid rgba(45,212,191,.3);border-radius:8px;'
                'padding:8px 16px;font-family:monospace;font-size:11px;font-weight:bold;'
                'text-decoration:none;letter-spacing:.5px;">NHP India &mdash; Health Portal</a>'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )

        # ── DOCTOR CHATBOT ────────────────────────────────────
        st.write("")
        with st.container(border=True):
            st.markdown('<span class="scap">Talk to AI Doctor &mdash; Describe Your Symptoms</span>', unsafe_allow_html=True)
            st.write("")
            st.markdown(
                '<div style="background:linear-gradient(135deg,rgba(45,212,191,.06),rgba(5,150,105,.04));'
                'border:1px solid rgba(45,212,191,.2);border-radius:10px;padding:12px 16px;'
                'font-family:monospace;font-size:11px;color:#fcd34d;letter-spacing:.5px;margin-bottom:14px;">'
                '&#9877; This AI doctor provides general health information only. It does NOT prescribe medication '
                'or replace a real physician. Always consult a licensed doctor for medical decisions.'
                '</div>',
                unsafe_allow_html=True
            )

            if "doctor_chat" not in st.session_state:
                st.session_state.doctor_chat = []
                st.session_state.doctor_chat.append((
                    "assistant",
                    f"Hello {st.session_state.p_name}! I'm your AI health assistant. "
                    f"Your X-ray screening result is **{'POSITIVE - Pneumonia Suspected' if is_pos else 'NEGATIVE - No Pneumonia Detected'}** "
                    f"with **{confidence:.1%} confidence**.\n\n"
                    f"To help me understand your situation better, could you tell me:\n"
                    f"1. What symptoms are you experiencing (fever, cough, breathlessness, chest pain, fatigue)?\n"
                    f"2. When did these symptoms start, and have they been getting better, worse, or staying the same?\n\n"
                    f"I'm here to help guide you and figure out how urgently you should see a doctor."
                ))

            doc_chat_container = st.container()
            with doc_chat_container:
                for role, msg in st.session_state.doctor_chat:
                    with st.chat_message(role, avatar="🩺" if role=="assistant" else "🧑"):
                        st.markdown(msg)

            if doc_q := st.chat_input("Describe your symptoms or ask a health question…", key="doctor_chat_input"):
                st.session_state.doctor_chat.append(("user", doc_q))
                doc_history = "\n".join(
                    f"{'DOCTOR' if r=='assistant' else 'PATIENT'}: {m}"
                    for r, m in st.session_state.doctor_chat[-12:]
                )
                doc_prompt = f"""You are a compassionate, knowledgeable AI doctor assistant inside PNEUMONIA.AI.

PATIENT DETAILS:
- Name: {st.session_state.p_name}
- Age: {st.session_state.p_age} years
- Gender: {st.session_state.p_gender}
- Location: {plocation}
- X-ray Result: {'PNEUMONIA SUSPECTED' if is_pos else 'NO PNEUMONIA DETECTED'}
- Confidence: {confidence:.1%}
- Pneumonia Type (user-entered): {ptype}
- Affected Side (user-entered): {pside}
- Grad-CAM Focus Region: {region}

YOUR ROLE: Conduct a structured, empathetic triage conversation. You are NOT diagnosing
and NOT prescribing — you are gathering information and helping the patient understand
how urgently they should seek in-person medical care.

TRIAGE CHECKLIST — work through these naturally over the conversation, one or two
questions at a time (do not interrogate the patient with a long list at once):
- Onset & duration: When did symptoms start? Sudden or gradual? Improving, stable, or worsening?
- Fever: Do they have a fever? If so, roughly how high, and have they measured their temperature?
- Cough: Dry or producing mucus/phlegm? What color is the phlegm, if any? Any blood?
- Breathing: Any shortness of breath, or breathing faster than normal? Does it happen at rest or only with activity?
- Chest symptoms: Any chest pain or tightness? Does it worsen with breathing or coughing?
- Energy & appetite: Unusual tiredness, weakness, confusion, or reduced appetite/fluid intake?
- Risk factors: Any existing conditions (asthma, COPD, heart disease, diabetes, immune issues), pregnancy, or age extremes (very young child or elderly)?
- Oxygen/color: If they have a pulse oximeter, what does it read? Any bluish tint to lips or fingertips?

RED-FLAG ESCALATION — if the patient reports ANY of the following, immediately and clearly
recommend urgent in-person/emergency care (hospital or urgent care, not "wait and see"):
- Severe or worsening shortness of breath, or breathing difficulty at rest
- Bluish or grey lips/face/fingertips
- Confusion, severe drowsiness, or difficulty staying awake
- Chest pain that is severe, persistent, or worsening
- High fever that won't come down, or fever with rigors/shaking chills
- Coughing up blood
- Signs of dehydration (very little urination, dizziness)
- The patient is a young child, elderly, pregnant, or has a chronic condition and symptoms are worsening

YOUR RULES:
1. Be warm, empathetic, and clear like a real doctor talking to a patient.
2. Ask 1-2 focused follow-up questions per turn from the triage checklist above — don't overwhelm the patient.
3. Briefly explain what the X-ray screening result means in simple, plain language when relevant.
4. NEVER prescribe specific medications, dosages, or brand names — not even over-the-counter ones.
5. Always recommend consulting a real physician for diagnosis and treatment.
6. If ANY red-flag symptom is mentioned, prioritize urgent care guidance over further questions.
7. Keep responses concise, 3 to 6 sentences max per reply.
8. You may mention general supportive self-care (rest, fluids, avoiding cold/smoky air, monitoring temperature) without naming any drug.

CONVERSATION SO FAR:
{doc_history}

PATIENT: {doc_q}
DOCTOR:"""

                with st.spinner("Doctor is responding…"):
                    doc_reply = _groq_call(doc_prompt)

                st.session_state.doctor_chat.append(("assistant", doc_reply))
                st.rerun()

            if len(st.session_state.doctor_chat) > 1:
                st.write("")
                if st.button("Clear Doctor Chat", use_container_width=True, key="clear_doc_chat"):
                    st.session_state.doctor_chat = []
                    st.rerun()

        st.write("")
        with st.container(border=True):
            st.markdown('<span class="scap">AI-Generated Narrative Summary</span>',unsafe_allow_html=True)
            st.write("")
            if st.session_state.diagnostic_context is None:
                with st.spinner("Generating AI summary…"):
                    st.session_state.diagnostic_context = groq_summary(
                        {"name":st.session_state.p_name,"age":st.session_state.p_age,
                         "gender":st.session_state.p_gender,"location":plocation},
                        prob, region, ptype, pside, is_pos)
            st.info(st.session_state.diagnostic_context)
            pdf=make_pdf({"name":st.session_state.p_name,"age":st.session_state.p_age,
                          "gender":st.session_state.p_gender,"location":plocation},
                         prob,region,ptype,pside,raw_img,overlay_m,loc_fig,is_pos)
            st.write("")
            ac1,ac2=st.columns(2)
            with ac1:
                st.download_button("Export PDF Report",data=pdf,
                    file_name=f"PneumoniaAI_{st.session_state.p_name.replace(' ','_')}.pdf",
                    mime="application/pdf",use_container_width=True)
            with ac2:
                if st.button("Reset Session",use_container_width=True):
                    st.session_state.pipeline_active=False
                    st.session_state.diagnostic_context=None
                    st.session_state.doctor_chat=[]
                    st.rerun()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(f"""
<div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
  {brand_svg(32)}
  <h3 style="font-family:sans-serif;font-weight:800;color:#fff;margin:0;font-size:15px;">Ask About This Tool</h3>
</div>
<p style="font-family:monospace;font-size:11px;color:#5eead4;margin:0 0 12px;">GENERAL INFO ONLY &mdash; NOT MEDICAL ADVICE</p>
""", unsafe_allow_html=True)
    st.write("---")
    for role,msg in st.session_state.chat_history:
        with st.chat_message(role): st.markdown(msg)
    if q:=st.chat_input("Ask how the tool works…"):
        with st.chat_message("user"): st.markdown(q)
        st.session_state.chat_history.append(("user",q))
        nq=q.lower()
        if any(w in nq for w in ("medicine","dosage","prescri","treat","cure","drug")):
            rep="I cannot provide medication or treatment recommendations. Please consult a licensed physician."
        else:
            ctx=None
            if st.session_state.pipeline_active and "p_name" in st.session_state:
                try:
                    ctx={"label":"PNEUMONIA SUSPECTED" if prob>0.5 else "NO PNEUMONIA DETECTED",
                         "confidence":f"{confidence:.1%}","focus":region,"ptype":ptype,"side":pside}
                except: pass
            rep=groq_chat(q,st.session_state.chat_history,ctx)
        with st.chat_message("assistant"): st.markdown(rep)
        st.session_state.chat_history.append(("assistant",rep))
