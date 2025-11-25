import streamlit as st
from PIL import Image
import numpy as np
import cv2
import easyocr
import io

# --------------------- CONFIGURACIÓN DE PÁGINA ---------------------
st.set_page_config(page_title="Conversor OCR", layout="centered", initial_sidebar_state="collapsed")

# --------------------- ESTADO INICIAL ---------------------
if "show_uploader" not in st.session_state:
    st.session_state.show_uploader = False

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

# --------------------- TÍTULO PRINCIPAL ---------------------
st.title("Conversor de Imagen a Texto")
st.caption("Convierte imágenes en texto fácilmente (PNG/JPG).")


# --------------------- FUNCIÓN PARA MOSTRAR BOTÓN INICIAL ---------------------
def show_upload_button():
    if st.button("Carga o copia tu imagen"):
        st.session_state.show_uploader = True


# --------------------- OCR SETUP ---------------------
@st.cache_resource(show_spinner=False)
def create_reader(langs):
    # Versión compatible con todas las versiones de EasyOCR
    return easyocr.Reader(langs, gpu=False)


def read_image_bytes(file) -> np.ndarray:
    img = Image.open(io.BytesIO(file)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# --------------------- PREPROCESAMIENTO MEJORADO ---------------------
def preprocess(img):
    # Escalar para mejorar nitidez
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Pasar a gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Quitar ruido
    denoised = cv2.medianBlur(gray, 3)

    # Mejorar contraste
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # Binarización estable
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh


def ocr_read(image_input, reader):
    result = reader.readtext(image_input)
    texts = [t[1] for t in result]
    final = "\n".join(texts)
    return final, result


# --------------------- INTERFAZ PRINCIPAL ---------------------
if not st.session_state.show_uploader:
    show_upload_button()

if st.session_state.show_uploader:
    uploaded = st.file_uploader("", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

    if uploaded:
        st.session_state.uploaded_file = uploaded
        st.session_state.show_uploader = False


# --------------------- PROCESO DEL OCR ---------------------
uploaded = st.session_state.uploaded_file

if uploaded:

    content = uploaded.read()
    img = read_image_bytes(content)

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
             caption="Imagen subida", use_column_width=True)

    # Idioma fijo: español
    langs = ["es"]
    reader = create_reader(langs)

    # Preprocesamiento siempre activo
    img_proc = preprocess(img)

    with st.spinner("Leyendo texto..."):
        text, raw = ocr_read(img_proc, reader)

    st.subheader("Texto detectado")

    if text.strip() == "":
        st.info("No se detectó texto.")
    else:
        st.code(text)

        # Descargar resultado
        st.download_button(
            "Descargar texto",
            text.encode("utf-8"),
            file_name=uploaded.name + "_ocr.txt"
        )

    # Botón reset
    if st.button("Hagámoslo de nuevo"):
        st.session_state.clear()
        st.experimental_rerun()

else:
    if st.session_state.show_uploader:
        st.info("Selecciona una imagen para comenzar.")
