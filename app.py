import streamlit as st
from PIL import Image
import numpy as np
import cv2
import easyocr
import io
import base64

st.set_page_config(page_title="Conversor OCR", layout="centered")

# -----------------------------------------------------------
# ESTADO INICIAL
# -----------------------------------------------------------
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None


# -----------------------------------------------------------
# TÍTULO
# -----------------------------------------------------------
st.title("Conversor de Imagen a Texto")
st.caption("Convierte imágenes en texto fácilmente (PNG/JPG).")


# -----------------------------------------------------------
# FUNCIÓN PARA ABRIR SELECTOR DE ARCHIVOS SIN MOSTRAR UPLOADER
# -----------------------------------------------------------
def custom_file_uploader():
    uploaded_file = st.file_uploader(
        "Selecciona una imagen",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed"
    )
    return uploaded_file


# -----------------------------------------------------------
# BOTÓN INICIAL — abrir selector sin mostrar uploader visual
# -----------------------------------------------------------
if st.session_state.uploaded_file is None:

    # botón que simula la carga sin mostrar uploader adelante
    if st.button("Carga o copia tu imagen"):

        # mostramos uploader, pero oculto
        st.session_state.show_hidden_uploader = True

    # si debe mostrarse el uploader oculto:
    if st.session_state.get("show_hidden_uploader", False):

        uploaded = custom_file_uploader()

        if uploaded:
            st.session_state.uploaded_file = uploaded
            st.session_state.show_hidden_uploader = False
            st.experimental_rerun()

else:
    # -----------------------------------------------------------
    # PROCESAR IMAGEN SUBIDA
    # -----------------------------------------------------------
    uploaded = st.session_state.uploaded_file
    content = uploaded.read()

    img = Image.open(io.BytesIO(content)).convert("RGB")
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    st.image(img, caption="Imagen subida", use_column_width=True)

    # -----------------------------------------------------------
    # OCR
    # -----------------------------------------------------------
    @st.cache_resource(show_spinner=False)
    def create_reader():
        return easyocr.Reader(["es"], gpu=False)

    reader = create_reader()

    # Preprocesamiento ligero
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    st.write("Procesando OCR...")

    text_result = reader.readtext(gray)
    detected = "\n".join([t[1] for t in text_result])

    st.subheader("Texto detectado")
    st.code(detected if detected.strip() else "No se detectó texto.")

    # Descargar
    st.download_button(
        "Descargar texto",
        detected.encode("utf-8"),
        file_name=uploaded.name + "_ocr.txt"
    )

    # -----------------------------------------------------------
    # BOTÓN PARA REINICIAR
    # -----------------------------------------------------------
    if st.button("Hagámoslo de nuevo"):
        st.session_state.clear()
        st.experimental_rerun()
