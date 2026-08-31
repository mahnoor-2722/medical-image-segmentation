# ============================================================
# Streamlit Web Application: Skin Lesion Segmentation
# Model: SegFormer (Vision Transformer) | PyTorch
# Author: Mahnoor (github.com/mahnoor-2722)
# ============================================================

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import streamlit as st
from PIL import Image
from transformers import SegformerForSemanticSegmentation
from huggingface_hub import hf_hub_download

# ─────────────────────────── PAGE CONFIG ─────────────────────────── #
st.set_page_config(
    page_title="Skin Lesion Segmentation AI",
    page_icon="🧠",
    layout="wide"
)

# ─────────────────────────── CONSTANTS ───────────────────────────── #
HF_REPO_ID = "mahnoor-2722/isic-segformer"
MODEL_FILENAME = "best_segformer_isic2018.pth"
IMG_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────── MODEL LOADER ─────────────────────────── #
@st.cache_resource
def load_segformer_model():
    """Load SegFormer architecture and pretrained weights from Hugging Face Hub"""
    try:
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0",
            num_labels=1,
            ignore_mismatched_sizes=True
        )
        model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=MODEL_FILENAME
        )
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# ─────────────────────────── HELPERS ─────────────────────────── #
def preprocess_image(pil_img):
    """Resize + Normalize input (ImageNet statistics)"""
    img = np.array(pil_img.convert("RGB"))
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm_img = (img_resized.astype(np.float32) / 255.0 - mean) / std
    tensor_img = torch.tensor(norm_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return img_resized, tensor_img

def predict_probability_map(model, tensor_img):
    """Run inference and return raw probability map"""
    with torch.no_grad():
        tensor_img = tensor_img.to(DEVICE)
        outputs = model(pixel_values=tensor_img)
        logits = F.interpolate(
            outputs.logits, size=(IMG_SIZE, IMG_SIZE),
            mode="bilinear", align_corners=False
        )
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    return probs

def compute_metrics(pred_mask, gt_mask_img):
    """Compute Dice and IoU with uploaded ground-truth mask"""
    gt_arr = np.array(gt_mask_img.convert("L"))
    gt_arr = cv2.resize(gt_arr, (IMG_SIZE, IMG_SIZE))
    gt_binary = (gt_arr > 127).astype(np.uint8)
    intersection = np.sum(pred_mask * gt_binary)
    total_sum = np.sum(pred_mask) + np.sum(gt_binary)
    union = total_sum - intersection
    dice = (2.0 * intersection + 1e-6) / (total_sum + 1e-6)
    iou = (intersection + 1e-6) / (union + 1e-6)
    return dice, iou, gt_binary

# ═══════════════════════════ UI LAYOUT ═══════════════════════════ #

st.title("🧠 Skin Lesion Segmentation — Vision Transformer (SegFormer)")
st.caption(
    "Trained on ISIC 2018 dermatoscopic images · Best with close-up lesion photos · "
    "Clear skin or non-dermoscopic photos may produce false segmentations."
)

# ─────── SIDEBAR ─────── #
st.sidebar.header("⚙️ Model Configuration")
st.sidebar.info(
    f"**Architecture:** SegFormer (MiT-B0)\n\n"
    f"**Framework:** PyTorch\n\n"
    f"**Hosting:** Hugging Face Hub\n\n"
    f"`{HF_REPO_ID}`\n\n"
    f"**Device:** `{DEVICE}`"
)

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Inference Threshold")
threshold = st.sidebar.slider(
    "Mask threshold",
    min_value=0.30, max_value=0.90, value=0.55, step=0.05
)
st.sidebar.caption("Higher = stricter lesion detection. Try increasing this on clear-skin images.")

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Benchmark Results")
st.sidebar.markdown(
    "| Model | Dice | IoU |\n"
    "|:------|:----:|:---:|\n"
    "| U-Net (TF) | 85.26% | 74.53% |\n"
    "| **SegFormer (PT)** | **90.32%** | **82.50%** |"
)

# ─────── LOAD MODEL ─────── #
with st.spinner("⏳ Loading SegFormer model from Hugging Face Hub..."):
    model = load_segformer_model()

if model is None:
    st.stop()

st.sidebar.success("✅ Model Loaded!")

# ─────── FILE UPLOADS ─────── #
col_a, col_b = st.columns(2)
with col_a:
    uploaded_file = st.file_uploader(
        "📷 Upload Dermatoscopic Skin Image",
        type=["jpg", "jpeg", "png"]
    )
with col_b:
    uploaded_gt = st.file_uploader(
        "🎭 (Optional) Upload Ground Truth Mask",
        type=["png", "jpg", "jpeg"]
    )

# ─────── PROCESS UPLOAD ─────── #
if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
    orig_img_resized, tensor_img = preprocess_image(input_image)

    with st.spinner("🔬 Analyzing image with SegFormer..."):
        prob_map = predict_probability_map(model, tensor_img)

    # Apply user-chosen threshold
    binary_mask = (prob_map > threshold).astype(np.uint8)
    lesion_pixels = int(binary_mask.sum())
    total_pixels = IMG_SIZE * IMG_SIZE
    lesion_ratio = (lesion_pixels / total_pixels) * 100.0

    # Confidence (mean probability inside the predicted region)
    if lesion_pixels > 0:
        confidence = float(prob_map[binary_mask == 1].mean())
    else:
        confidence = 0.0

    # Heuristics for possible false detection
    NO_LESION_DETECTED = lesion_pixels == 0
    LOW_CONFIDENCE = confidence < 0.70
    OVERSIZE_MASK = lesion_ratio > 45.0        # clear skin often blobs > 45%
    SUSPICIOUS = NO_LESION_DETECTED or LOW_CONFIDENCE or OVERSIZE_MASK

    # Green overlay for visualization
    overlay = orig_img_resized.copy()
    overlay[binary_mask == 1] = (
        overlay[binary_mask == 1] * 0.5 + np.array([0, 255, 0]) * 0.5
    )

    st.markdown("---")
    st.subheader("📊 Segmentation Results")

    # Smart status banner
    if NO_LESION_DETECTED:
        st.warning(
            "⚠️ **No lesion detected at current threshold.** "
            "Try lowering the mask threshold in the sidebar, or upload a "
            "dermatoscopic close-up of an actual lesion."
        )
    elif OVERSIZE_MASK or LOW_CONFIDENCE:
        st.warning(
            f"⚠️ **Possible false detection** (confidence ≈ {confidence*100:.1f}%, "
            f"area ≈ {lesion_ratio:.1f}%).\n\n"
            "This model was trained specifically on **ISIC 2018 dermatoscopic lesion images**. "
            "On clear/normal skin or non-dermoscopic photos, it may produce inaccurate masks. "
            "Please use a proper dermatoscopic image, or raise the threshold slider."
        )
    else:
        st.success(
            f"✅ **Lesion region detected** with mean confidence **{confidence*100:.1f}%**."
        )

    # Result columns
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.image(orig_img_resized, caption="1. Input Image", use_container_width=True)
    with c2:
        st.image(binary_mask * 255, caption="2. Predicted Binary Mask", use_container_width=True)
    with c3:
        st.image(prob_map, caption="3. Probability Heatmap", use_container_width=True, clamp=True)
    with c4:
        st.image(overlay, caption="4. Lesion Boundary Overlay", use_container_width=True)

    st.markdown("---")

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Affected Area", f"{lesion_ratio:.2f}%")
    m2.metric("Mean Confidence", f"{confidence*100:.1f}%")
    m3.metric("Threshold Used", f"{threshold:.2f}")
    m4.metric("Resolution", f"{IMG_SIZE} × {IMG_SIZE}")

    # Ground-truth comparison (optional)
    if uploaded_gt is not None:
        gt_img = Image.open(uploaded_gt)
        dice, iou, _ = compute_metrics(binary_mask, gt_img)
        st.markdown("---")
        st.subheader("🎯 Ground Truth Comparison")
        g1, g2 = st.columns(2)
        g1.metric("Dice Coefficient (Overlap)", f"{dice*100:.2f}%")
        g2.metric("IoU (Jaccard Index)", f"{iou*100:.2f}%")

    st.markdown("---")
    st.info(
        "📌 **Disclaimer:** This is a research demonstration built for the ISIC 2018 "
        "dermatoscopic dataset. It is **not** a medical diagnostic tool and "
        "should not be used for clinical decisions. Best results are obtained on "
        "close-up dermatoscopic lesion images."
    )

else:
    st.info("👆 Please upload a dermatoscopic image above to test the segmentation model.")
    
    with st.expander("📖 How to Use This Demo"):
        st.markdown("""
        1. **Upload a dermatoscopic image** — ideally a close-up of a real skin lesion (JPG/PNG).
        2. *(Optional)* Upload the ground-truth binary mask to compute **Dice/IoU** scores.
        3. Adjust the **mask threshold** slider in the sidebar if needed.
        4. View 4 outputs: original, predicted mask, probability heatmap, and green overlay.
        
        **Note:** This model was trained specifically on ISIC 2018 lesion images. It may 
        produce inaccurate masks on non-dermoscopic photos, clear skin, or unrelated content.
        """)

# ─────── FOOTER ─────── #
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray; font-size:0.85em;'>"
    "Built by <b>Mahnoor</b> · Vision Transformer (SegFormer MiT-B0) on ISIC 2018 · "
    "<a href='https://github.com/mahnoor-2722/medical-image-segmentation' target='_blank'>GitHub</a> · "
    "<a href='https://huggingface.co/mahnoor-2722/isic-segformer' target='_blank'>Hugging Face</a>"
    "</div>",
    unsafe_allow_html=True
)