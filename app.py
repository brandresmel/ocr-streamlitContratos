import streamlit as st
from PIL import Image
import numpy as np
import cv2
import easyocr
import io

# --------------------- CONFIGURACIÓN ---------------------
st.set_page_config(page_title="Conversor OCR", layout="centered")

# Estado
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None


# --------------------- JACKPOT: FILE UPLOADER OCULTO ---------------------
# Creamos un uploader escondido completamente
uploaded = st.file_uploader(
    "hidden uploader",
    type=["png", "jpg", "jpeg"],
    label_visibility="collapsed",
    key="hidden_uploader"
)

# Lo escondemos via HTML/CSS
hide_uploader_style = """
<style>
#hidden_uploader {display:none;}
section[data-testid="stFileUploadDropzone"] {display:none !important;}
</style>
"""
st.markdown(hide_uploader_style, unsafe_allow_html=True)


# --------------------- BOTÓN PARA ABRIR EXPLORADOR ---------------------
st.title("Conversor de Imagen a Texto")
st.caption("Convierte imágenes en texto fácilmente (PNG/JPG).")

st.write("")  
launch = st.button("Carga o copia tu imagen", use_container_width=True)

# Si hacen clic → disparamos el uploader oculto con JS
if launch:
    js = """
    <script>
        const inputs = window.parent.document.querySelectorAll('input[type="file"]');
        for (let i = 0; i < inputs.length; i++) {
            inputs[i].click();
        }
    </script>
    """
    st.markdown(js, unsafe_allow_html=True)


# Si el uploader capturó un archivo, guardarlo
if uploaded is not None:
    st.session_state.uploaded_file = uploaded


# --------------------- PROCESO OCR ---------------------
uploaded = st.session_state.uploaded_file

def read_image_bytes(file) -> np.ndarray:
    img = Image.open(io.BytesIO(file)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 15
    )
    return thresh

@st.cache_resource(show_spinner=False)
def create_reader():
    return easyocr.Reader(["es"], gpu=False)

reader = create_reader()

if uploaded:
    content = uploaded.read()
    img = read_image_bytes(content)

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
             caption="Imagen subida", use_column_width=True)

    img_proc = preprocess(img)

    with st.spinner("Leyendo texto..."):
        result = reader.readtext(img_proc)
        text = "\n".join([t[1] for t in result])

    st.subheader("Texto detectado")
    st.code(text if text.strip() else "No se detectó texto.")

    st.download_button(
        "Descargar texto",
        text.encode("utf-8"),
        file_name=uploaded.name + "_ocr.txt"
    )

    if st.button("Hagámoslo de nuevo"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()
