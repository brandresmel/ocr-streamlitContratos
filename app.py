import streamlit as st
from PIL import Image
import numpy as np
import cv2
import easyocr
import io

st.set_page_config(page_title="Lector OCR", layout="centered")
st.title("Carga o copia tu imagen")
st.caption("Sube una imagen con texto y obtén el texto detectado. Puedes repetir las veces que quieras.")

with st.sidebar:
    st.header("Opciones")
    lang_choice = st.selectbox("Idioma", ["es", "en", "es+en"], index=0)
    preproc = st.checkbox("Mejorar lectura (preprocesamiento)", value=True)
    clear_button = st.button("Hagámoslo de nuevo")

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

uploaded = st.file_uploader("Selecciona una imagen (PNG/JPG)", type=["png","jpg","jpeg"])

if clear_button:
    st.experimental_rerun()

if uploaded:
    content = uploaded.read()
    img = read_image_bytes(content)

    st.image(cv2.cvtColor(img,cv2.COLOR_BGR2RGB),caption="Imagen subida",use_column_width=True)

    langs = ["es"] if lang_choice=="es" else ["en"] if lang_choice=="en" else ["es","en"]

    reader = create_reader(langs)

    if preproc:
        img_proc = preprocess(img)
        st.image(img_proc, caption="Preprocesada", use_column_width=True)
        img_for_ocr = img_proc
    else:
        img_for_ocr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with st.spinner("Leyendo texto..."):
        text, raw = ocr_read(img_for_ocr, reader)

    st.subheader("Texto detectado")
    if text.strip()=="":
        st.info("No se detectó texto.")
    else:
        st.code(text)

        st.download_button("Descargar texto", text.encode("utf-8"),
                           file_name=uploaded.name+"_ocr.txt")

    if st.checkbox("Ver detalles (debug)"):
        st.write(raw)

else:
    st.info("Sube una imagen para comenzar.")
