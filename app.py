import streamlit as st
from PIL import Image
import numpy as np
import cv2
import pytesseract
from paddleocr import PaddleOCR
import io
from textblob import TextBlob

# ---------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA
# ---------------------------------------------------------
st.set_page_config(page_title="Conversor OCR", layout="centered")

st.title("Conversor de Imagen a Texto")
st.caption("Convierte imágenes con texto en texto editable usando OCR mejorado.")

# Estado inicial
if "show_uploader" not in st.session_state:
    st.session_state.show_uploader = True

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None


# ---------------------------------------------------------
# PREPROCESAMIENTO AVANZADO
# ---------------------------------------------------------
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Aumentar contraste
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Reducción de ruido
    denoise = cv2.fastNlMeansDenoising(enhanced, h=9)

    # Binarización adaptativa
    thresh = cv2.adaptiveThreshold(
        denoise, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35, 11
    )
    return thresh


# ---------------------------------------------------------
# OCR ENGINE (Paddle + fallback a Tesseract)
# ---------------------------------------------------------
@st.cache_resource
def load_ocr():
    return PaddleOCR(
        use_angle_cls=True,
        lang="es",
        use_gpu=False,
        rec=True,
        det=True
    )

ocr_engine = load_ocr()


def extract_text_with_paddle(img):
    result = ocr_engine.ocr(img, cls=True)
    lines = []

    for block in result:
        for piece in block:
            lines.append(piece[1][0])

    return "\n".join(lines)


def extract_with_tesseract(img):
    return pytesseract.image_to_string(img, lang="spa")


# ---------------------------------------------------------
# INTERFAZ
# ---------------------------------------------------------
# Botón inicial
if st.session_state.show_uploader:
    if st.button("Carga o copia tu imagen"):
        st.session_state.show_uploader = True
        st.session_state.uploaded_file = None

    uploaded = st.file_uploader("", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

    if uploaded:
        st.session_state.uploaded_file = uploaded
        st.session_state.show_uploader = False
        st.rerun()

# Si ya hay imagen
uploaded = st.session_state.uploaded_file

if uploaded:
    img_bytes = uploaded.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    st.image(img_np, caption="Imagen cargada", use_column_width=True)

    # Preprocesamiento
    preproc_img = preprocess_image(img_np)

    with st.spinner("Leyendo texto..."):
        try:
            text = extract_text_with_paddle(preproc_img)

            if text.strip() == "":
                text = extract_text_with_paddle(img_np)

        except Exception:
            text = extract_with_tesseract(preproc_img)

    # Corrección básica con TextBlob
    corrected = text

    st.subheader("Texto detectado (OCR mejorado):")
    st.code(corrected)

    st.download_button(
        "Descargar texto",
        corrected.encode("utf-8"),
        file_name="resultado_ocr.txt"
    )

    # Botón para reiniciar
    if st.button("Hagámoslo de nuevo"):
        st.session_state.clear()
        st.rerun()
