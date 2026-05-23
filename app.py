import os
import tempfile

import streamlit as st

from src.model_utils import load_model
from src.predict import predict_audio


CHECKPOINT_PATH = "checkpoints/best.pt"

st.set_page_config(
    page_title="VCCC - Audio Deepfake Detection",
    page_icon="🎙️",
    layout="centered"
)

st.title("VCCC - Audio Deepfake Detection")
st.write(
    "Aplikasi deteksi audio asli atau deepfake menggunakan model AASIST3-KAN."
)

uploaded_file = st.file_uploader(
    "Upload file audio",
    type=["wav", "flac", "mp3"]
)

if uploaded_file is not None:
    st.audio(uploaded_file)

    st.write("**Informasi File**")
    st.write(f"Nama file: `{uploaded_file.name}`")
    st.write(f"Ukuran file: `{uploaded_file.size / 1024:.2f} KB`")

    if st.button("Deteksi Audio"):
        suffix = os.path.splitext(uploaded_file.name)[1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            with st.spinner("Memuat model dan memproses audio..."):
                model = load_model(CHECKPOINT_PATH)
                result = predict_audio(model, tmp_path)

            st.subheader("Hasil Deteksi")

            if result["class_id"] == 0:
                st.error(f"Label Prediksi: {result['label']}")
                st.warning("Audio terindikasi sebagai hasil manipulasi atau deepfake.")
            else:
                st.success(f"Label Prediksi: {result['label']}")
                st.info("Audio terindikasi sebagai audio asli atau bona fide.")

            st.write(f"Confidence: **{result['confidence']:.2f}%**")

            with st.expander("Detail Probabilitas"):
                st.write(result["probabilities"])

        except Exception as e:
            st.error("Terjadi error saat proses deteksi.")
            st.exception(e)

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)