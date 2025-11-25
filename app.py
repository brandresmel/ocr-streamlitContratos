import streamlit as st
import cv2
import numpy as np
from paddleocr import PaddleOCR
import pytesseract
from PIL import Image
import io
import re
import textblob

# ---------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA
# ---------------------------------------------------------
st.set_page_config(
    page_title="Conversor de Imagen a Texto",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Ocultar el widget original de file_uploader
hide_upload_style = """
<style>
[data-testid="stFileUploader"] {
    display: none;
}
</style>
"""
st.markdown(hide_upload_style, unsafe_allow_html=True)

# ---------------------------------------------------------
# ESTADO
# ---------------------------------------------------------
if "show_uploader" not in st.session_state:
    st.session_state.show_uploader = False

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None


# ---------------------------------------------------------
# INTERFAZ
# ---------------------------------------------------------
st.title("Conversor de Imagen a Texto")
st.caption("Convierte imágenes (PNG/JPG) en texto con alta precisión.")

# Botón inicial
if not st.session_state.show_uploader and st.session_state.uploaded_file is None:
    if st.button("Carga o copia tu imagen"):
        st.session_state.show_uploader = True


# File uploader oculto pero activado por el botón
if st.session_state.show_uploader:
    uploaded = st.file_uploader(" ", type=["png", "jpg", "jpeg"])
    if uploaded:
        st.session_state.uploaded_file = uploaded
        st.session_state.show_uploader = False


# ---------------------------------------------------------
# PIPELINE: SUPER-RESOLUCIÓN + PREPROCESO + DETECCIÓN + OCR
# ---------------------------------------------------------
def upscale_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def enhance_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    denoise = cv2.fastNlMeansDenoising(gray, h=10)
    thresh = cv2.adaptiveThreshold(
        denoise, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        41, 8
    )
    return thresh

# Detector PaddleOCR (solo detección)
paddle = PaddleOCR(use_angle_cls=True, lang="es", rec=False)

def ocr_tesseract(crop):
    config = "--oem 3 --psm 6 -l spa"
    return pytesseract.image_to_string(crop, config=config).strip()

def correct_text(text):
    def safe_correct(word):
        if re.fullmatch(r"[0-9.,]+", word):
            return word
        try:
            return str(textblob.TextBlob(word).correct())
        except:
            return word
    return " ".join(safe_correct(w) for w in text.split())


# ---------------------------------------------------------
# PROCESAR IMAGEN
# ---------------------------------------------------------
uploaded = st.session_state.uploaded_file

if uploaded:
    content = uploaded.read()

    img = Image.open(io.BytesIO(content)).convert("RGB")
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
             caption="Imagen subida",
             use_column_width=True)

    # Pipeline
    img = upscale_image(img)
    proc = enhance_image(img)

    with st.spinner("Leyendo texto con precisión mejorada..."):
        detections = paddle.ocr(proc, cls=True)

        final_lines = []
        for block in detections:
            for det in block:
                box = np.array(det[0]).astype(int)
                x1, y1 = box[:,0].min(), box[:,1].min()
                x2, y2 = box[:,0].max(), box[:,1].max()
                crop = proc[y1:y2, x1:x2]

                txt = ocr_tesseract(crop)
                if txt:
                    final_lines.append(txt)

        raw_text = "\n".join(final_lines)
        corrected = correct_text(raw_text)

    st.subheader("Texto detectado")
    st.code(corrected)

    st.download_button(
        "Descargar texto",
        corrected.encode("utf-8"),
        file_name=uploaded.name + "_ocr.txt"
    )

    if st.button("Hagámoslo de nuevo"):
        st.session_state.clear()
        st.rerun()

else:
    if st.session_state.show_uploader:
        st.info("Selecciona una imagen para comenzar.")
