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


# --------------------- FUNCIÓN PARA MOSTRAR UPLOADER ---------------------
def show_upload_button():
    if st.button("Carga o copia tu imagen"):
        st.session_state.show_uploader = True


# --------------------- OCR SETUP ---------------------
@st.cache_resource(show_spinner=False)
def create_reader(langs):
    return easyocr.Reader(langs, gpu=False)


def read_image_bytes(file) -> np.ndarray:
    img = Image.open(io.BytesIO(file)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    thresh = cv2.adaptiveThreshold(denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 15)
    return thresh


def ocr_read(image_input, reader):
    result = reader.readtext(image_input)
    texts = [t[1] for t in result]
    final = "\n".join(texts)
    return final, result


# --------------------- INTERFAZ PRINCIPAL ---------------------
# Mostrar botón inicial solo si aún no se mostró el uploader
if not st.session_state.show_uploader:
    show_upload_button()

# Cuando el botón fue presionado, mostrar uploader
if st.session_state.show_uploader:
    uploaded = st.file_uploader("", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

    if uploaded:
        st.session_state.uploaded_file = uploaded
        st.session_state.show_uploader = False  # ocultar uploader después de subir


# --------------------- PROCESO DEL OCR ---------------------
uploaded = st.session_state.uploaded_file

if uploaded:
    content = uploaded.read()
    img = read_image_bytes(content)

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
             caption="Imagen subida", use_column_width=True)

    # Idiomas fijos (sin sidebar)
    langs = ["es"]

    reader = create_reader(langs)

    # Preprocesar siempre (ya que quitamos la opción)
    img_proc = preprocess(img)

    with st.spinner("Leyendo texto..."):
        text, raw = ocr_read(img_proc, reader)

    st.subheader("Texto detectado")

    if text.strip() == "":
        st.info("No se detectó texto.")
    else:
        st.code(text)

        st.download_button("Descargar texto",
                           text.encode("utf-8"),
                           file_name=uploaded.name + "_ocr.txt")

    # ---------------- BOTÓN HAGÁMOSLO DE NUEVO ----------------
    if st.button("Hagámoslo de nuevo"):
        st.session_state.clear()
        st.experimental_rerun()

else:
    if st.session_state.show_uploader:
        st.info("Selecciona una imagen para comenzar.")
