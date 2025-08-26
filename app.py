import streamlit as st
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from utils.preprocessing import preprocess_signature
from utils.inference import predict_user, verify_signature

st.set_page_config(page_title="手写签名识别系统", layout="centered")

st.markdown("""
    <h1 style='text-align: center; color: #2C3E50;'>🖋️ 手写签名识别系统</h1>
    <p style='text-align: center; font-size: 16px; color: gray;'>上传签名图片，系统将自动识别归属或判断真伪</p>
""", unsafe_allow_html=True)

st.markdown("---")

# 上传签名图像
uploaded_file = st.file_uploader("📤 请上传需要识别的签名图像（支持 PNG/JPG 格式）", type=["png", "jpg", "jpeg"], label_visibility="visible")

# 功能选择
mode = st.radio("请选择识别功能：", ["签名归属判断（识别签名属于谁）", "签名真伪验证（判断签名是否伪造）"], horizontal=True)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    img_array = np.array(image)
    processed_img = preprocess_signature(img_array)

    st.markdown("#### 🖼️ 预处理后图像预览")
    st.image(processed_img, caption="处理后的签名图像", width=320)
    st.markdown("---")

    if mode.startswith("签名归属判断"):
        st.markdown("### 🧠 系统识别结果：")
        user_label, confidence = predict_user(processed_img)
        st.markdown(f"""
            <div style='background-color:#E8F5E9;padding:15px;border-radius:6px;font-size:17px'>
                🧑‍💼 <b>预测归属用户编号：</b> {user_label} <br>
                📊 <b>识别置信度：</b> {confidence:.2f}
            </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("### 🔍 系统真伪验证")
        with st.expander("📄 上传参考签名图像（真实签名，用于比对）"):
            ref_file = st.file_uploader("请上传同一人的真签图像（支持 PNG/JPG 格式）", type=["png", "jpg", "jpeg"], key="ref", label_visibility="visible")

        threshold = 2.480

        if ref_file is not None:
            ref_img = Image.open(ref_file).convert("L")
            ref_array = np.array(ref_img)
            ref_processed = preprocess_signature(ref_array)

            similarity, _ = verify_signature(processed_img, ref_processed, threshold)

            st.markdown("<hr style='margin-top:15px;margin-bottom:15px;'>", unsafe_allow_html=True)
            st.markdown("**🔎 系统判定结果：**")

            if similarity < threshold:
                st.success("✅ 该签名为真签，与参考图像一致")
            else:
                st.error("❌ 该签名可能为伪造，与参考图像差异较大")

            st.caption("该判断基于图像特征的相似度分析，系统已使用优化后的判断标准。")
