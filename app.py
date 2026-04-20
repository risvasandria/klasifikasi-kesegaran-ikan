import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort

# =============================================
# KONFIGURASI MODEL
# =============================================
MODEL_PATHS = {
    "CNN": "models/cnn_model.onnx",
    "EfficientNet": "models/efficientnet_model.onnx",
    "ResNet50": "models/resnet50_model.onnx",
}

# Nama kelas sesuai dataset
CLASS_NAMES = ["Busuk ", "Fresh", "Semi Fresh "]

IMG_SIZE = (224, 224)

# =============================================
# EMOJI PER KELAS
# =============================================
CLASS_EMOJI = {
    "Busuk": "🟤",
    "Fresh": "🟢",
    "Semi Fresh": "🟡",
}

# =============================================
# FUNGSI LOAD MODEL ONNX
# =============================================
@st.cache_resource
def load_onnx_model(model_path):
    session = ort.InferenceSession(model_path)
    return session

def predict_onnx(session, img_array):
    input_name = session.get_inputs()[0].name
    result = session.run(None, {input_name: img_array})
    return result[0][0]

def preprocess_image(image):
    img = image.convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# =============================================
# TAMPILAN APLIKASI STREAMLIT
# =============================================
st.set_page_config(
    page_title="Klasifikasi Kesegaran Ikan",
    page_icon="🐟",
    layout="centered"
)

st.title("🐟 Klasifikasi Kesegaran Ikan")
st.markdown("Upload foto ikan, sistem akan mendeteksi apakah **Busuk**, **Fresh**, atau **Semi Fresh**.")

# --- Sidebar ---
st.sidebar.header("⚙️ Pengaturan")
selected_model_name = st.sidebar.selectbox(
    "Pilih Model:",
    list(MODEL_PATHS.keys())
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Kelas yang dikenali:**")
for cls in CLASS_NAMES:
    emoji = CLASS_EMOJI.get(cls, "⚪")
    st.sidebar.markdown(f"{emoji} {cls}")

# --- Upload Gambar ---
uploaded_file = st.file_uploader(
    "📁 Upload Gambar Ikan (JPG / PNG / JPEG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼️ Gambar yang Diupload")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("📊 Hasil Prediksi")
        with st.spinner(f"Sedang menganalisis dengan {selected_model_name}..."):
            try:
                session = load_onnx_model(MODEL_PATHS[selected_model_name])
                img_array = preprocess_image(image)
                predictions = predict_onnx(session, img_array)

                predicted_index = np.argmax(predictions)
                predicted_class = CLASS_NAMES[predicted_index]
                confidence = float(predictions[predicted_index]) * 100
                emoji = CLASS_EMOJI.get(predicted_class, "⚪")

                st.success(f"{emoji} **{predicted_class}**")
                st.metric("Tingkat Keyakinan", f"{confidence:.2f}%")

                st.markdown("---")
                st.markdown("**Probabilitas per kelas:**")
                for i, cls in enumerate(CLASS_NAMES):
                    prob = float(predictions[i]) * 100
                    em = CLASS_EMOJI.get(cls, "⚪")
                    st.progress(int(prob), text=f"{em} {cls}: {prob:.2f}%")

            except FileNotFoundError:
                st.error(
                    f"❌ File model tidak ditemukan!\n\n"
                    "Pastikan folder `models/` berisi file `.onnx` di GitHub."
                )
            except Exception as e:
                st.error(f"❌ Terjadi error: {str(e)}")
else:
    st.info("👆 Silakan upload gambar ikan terlebih dahulu.")
    st.markdown("---")
    st.markdown("### 📖 Panduan Singkat")
    st.markdown("""
    1. Pilih model di **sidebar kiri** (CNN, EfficientNet, atau ResNet50)
    2. Klik tombol **Browse files** di atas
    3. Upload foto ikan kamu
    4. Hasil klasifikasi akan langsung muncul!
    """)

# --- Footer ---
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray;'>Dibuat dengan ❤️ menggunakan Streamlit & ONNX Runtime</div>",
    unsafe_allow_html=True
)