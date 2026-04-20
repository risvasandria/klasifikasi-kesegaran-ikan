import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# =============================================
# KONFIGURASI MODEL
# =============================================
MODEL_PATHS = {
    "CNN": "models/cnn_model.tflite",
    "EfficientNet": "models/efficientnet_model.tflite",
    "ResNet50": "models/resnet50_model.tflite",
}

# Nama kelas sesuai dataset (urutan sama seperti saat training)
CLASS_NAMES = ["Busuk", "Fresh", "Semi Fresh"]

IMG_SIZE = (224, 224)

# =============================================
# FUNGSI LOAD MODEL TFLITE
# =============================================
@st.cache_resource
def load_tflite_model(model_path):
    """Load model TFLite dan siapkan interpreter-nya"""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def predict_tflite(interpreter, img_array):
    """Jalankan prediksi menggunakan model TFLite"""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_data = img_array.astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0]

def preprocess_image(image):
    """Ubah gambar yang diupload menjadi format yang bisa dibaca model"""
    img = image.convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# =============================================
# EMOJI PER KELAS
# =============================================
CLASS_EMOJI = {
    "Busuk": "🟤",
    "Fresh": "🟢",
    "Semi Fresh": "🟡",
}

# =============================================
# TAMPILAN APLIKASI STREAMLIT
# =============================================
st.set_page_config(
    page_title="Klasifikasi Kesegaran - Deep Learning",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 Klasifikasi Kesegaran Bahan")
st.markdown("Upload foto bahan makanan, lalu sistem akan mendeteksi apakah **Busuk**, **Fresh**, atau **Semi Fresh**.")

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
    "📁 Upload Gambar (JPG / PNG / JPEG)",
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
                # Load model yang dipilih
                interpreter = load_tflite_model(MODEL_PATHS[selected_model_name])

                # Proses gambar & prediksi
                img_array = preprocess_image(image)
                predictions = predict_tflite(interpreter, img_array)

                # Ambil hasil terbaik
                predicted_index = np.argmax(predictions)
                predicted_class = CLASS_NAMES[predicted_index]
                confidence = predictions[predicted_index] * 100
                emoji = CLASS_EMOJI.get(predicted_class, "⚪")

                # Tampilkan hasil utama
                st.success(f"{emoji} **{predicted_class}**")
                st.metric("Tingkat Keyakinan", f"{confidence:.2f}%")

                # Tampilkan probabilitas semua kelas
                st.markdown("---")
                st.markdown("**Probabilitas per kelas:**")
                for i, cls in enumerate(CLASS_NAMES):
                    prob = float(predictions[i]) * 100
                    em = CLASS_EMOJI.get(cls, "⚪")
                    st.progress(int(prob), text=f"{em} {cls}: {prob:.2f}%")

            except FileNotFoundError:
                st.error(
                    f"❌ File model `{MODEL_PATHS[selected_model_name]}` tidak ditemukan!\n\n"
                    "Pastikan folder `models/` sudah ada di GitHub dan berisi file `.tflite`."
                )
            except Exception as e:
                st.error(f"❌ Terjadi error: {str(e)}")

else:
    st.info("👆 Silakan upload gambar terlebih dahulu untuk memulai klasifikasi.")

    st.markdown("---")
    st.markdown("### 📖 Panduan Singkat")
    st.markdown("""
    1. Pilih model di **sidebar kiri** (CNN, EfficientNet, atau ResNet50)
    2. Klik tombol **Browse files** di atas
    3. Upload foto bahan makanan kamu
    4. Hasil klasifikasi akan langsung muncul!
    """)

# --- Footer ---
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray;'>Dibuat dengan ❤️ menggunakan Streamlit & TensorFlow Lite</div>",
    unsafe_allow_html=True
)
