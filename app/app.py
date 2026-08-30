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

# Page Configuration
st.set_page_config(
    page_title="Skin Lesion Segmentation AI",
    page_icon="🧠",
    layout="wide"
)

# Constants
HF_REPO_ID = "mahnoor-2722/isic-segformer"
MODEL_FILENAME = "best_segformer_isic2018.pth"
IMG_SIZE = 256

# Device selection (CPU for web deployment, GPU if available)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_segformer_model():
    """Load SegFormer model architecture and download weights from HF Hub"""
    try:
        # Load base architecture
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0",
            num_labels=1,
            ignore_mismatched_sizes=True
        )
        
        # Download weights from HF
        model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=MODEL_FILENAME
        )
        
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model from Hugging Face: {e}")
        return None

def preprocess_image(pil_img):
    """Resize and normalize input image according to ImageNet standards"""
    img = np.array(pil_img.convert("RGB"))
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Normalize (ImageNet mean & std)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    norm_img = (img_resized.astype(np.float32) / 255.0 - mean) / std
    
    # Convert to Tensor (1, 3, 256, 256)
    tensor_img = torch.tensor(norm_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return img_resized, tensor_img

def predict_mask(model, tensor_img):
    """Run inference and return binary mask and probability map"""
    with torch.no_grad():
        tensor_img = tensor_img.to(DEVICE)
        outputs = model(pixel_values=tensor_img)
        logits = F.interpolate(outputs.logits, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        binary_mask = (probs > 0.5).astype(np.uint8)
    return binary_mask, probs

def compute_metrics(pred_mask, gt_mask_img):
    """Compute Dice and IoU if ground truth mask is uploaded"""
    gt_arr = np.array(gt_mask_img.convert("L"))
    gt_arr = cv2.resize(gt_arr, (IMG_SIZE, IMG_SIZE))
    gt_binary = (gt_arr > 127).astype(np.uint8)
    
    intersection = np.sum(pred_mask * gt_binary)
    total_sum = np.sum(pred_mask) + np.sum(gt_binary)
    union = total_sum - intersection
    
    dice = (2.0 * intersection + 1e-6) / (total_sum + 1e-6)
    iou = (intersection + 1e-6) / (union + 1e-6)
    return dice, iou, gt_binary

# ==================== STREAMLIT UI ==================== #

st.title("🧠 Medical Image Segmentation: Vision Transformer (SegFormer)")
st.markdown("""
This application uses a **SegFormer (MiT-B0)** Vision Transformer model fine-tuned on the **ISIC 2018 dataset** 
to perform pixel-level boundary segmentation of skin lesions.
""")

st.sidebar.header("⚙️ Model Configuration")
st.sidebar.info(f"**Architecture**: SegFormer (MiT-B0)\n\n**Framework**: PyTorch\n\n**Hosting**: Hugging Face Hub (`{HF_REPO_ID}`)\n\n**Device**: `{DEVICE}`")

# Load Model
with st.spinner("Loading SegFormer model from Hugging Face Hub..."):
    model = load_segformer_model()

if model is None:
    st.stop()

st.sidebar.success("✅ Model Loaded Successfully!")

# Main Layout
col_upload, col_gt = st.columns(2)

with col_upload:
    uploaded_file = st.file_uploader("📷 Upload Dermatoscopic Skin Image", type=["jpg", "jpeg", "png"])

with col_gt:
    uploaded_gt = st.file_uploader("🎭 (Optional) Upload Ground Truth Mask", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read Image
    input_image = Image.open(uploaded_file)
    orig_img_resized, tensor_img = preprocess_image(input_image)
    
    # Run Inference
    with st.spinner("Segmenting lesion boundaries..."):
        binary_mask, prob_map = predict_mask(model, tensor_img)
    
    # Create Green Overlay
    overlay = orig_img_resized.copy()
    overlay[binary_mask == 1] = overlay[binary_mask == 1] * 0.5 + np.array([0, 255, 0]) * 0.5
    
    # Compute Statistics
    lesion_pixels = np.sum(binary_mask)
    total_pixels = IMG_SIZE * IMG_SIZE
    lesion_ratio = (lesion_pixels / total_pixels) * 100
    
    st.markdown("---")
    st.subheader("📊 Segmentation Results & Visual Overlay")
    
    res_col1, res_col2, res_col3, res_col4 = st.columns(4)
    
    with res_col1:
        st.image(orig_img_resized, caption="1. Input Image", use_container_width=True)
    with res_col2:
        st.image(binary_mask * 255, caption="2. Predicted Binary Mask", use_container_width=True)
    with res_col3:
        st.image(prob_map, caption="3. Probability Heatmap", use_container_width=True)
    with res_col4:
        st.image(overlay, caption="4. Lesion Boundary Overlay", use_container_width=True)
        
    st.markdown("---")
    
    # Display KPIs
    stat_col1, stat_col2, stat_col3 = st.columns(3)
    
    with stat_col1:
        st.metric("Affected Lesion Area", f"{lesion_ratio:.2f}% of image")
    with stat_col2:
        st.metric("Resolution Processed", f"{IMG_SIZE} x {IMG_SIZE} px")
    with stat_col3:
        if uploaded_gt is not None:
            gt_img = Image.open(uploaded_gt)
            dice, iou, _ = compute_metrics(binary_mask, gt_img)
            st.metric("Dice Score (Overlap)", f"{dice*100:.2f}%")
        else:
            st.metric("Model Architecture", "SegFormer MiT-B0")

    if uploaded_gt is not None:
        st.success(f"🎯 Real-time Ground Truth Comparison | **Dice**: {dice*100:.2f}% | **IoU**: {iou*100:.2f}%")

else:
    st.info("👆 Please upload a dermatoscopic image above to test the segmentation AI.")