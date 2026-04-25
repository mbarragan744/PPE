import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import tempfile
import os
from ultralytics import YOLO

# Configuración
st.set_page_config(page_title="Detector YOLO", layout="centered")
st.title("🧠 Detección de objetos con YOLO (best.pt)")

# Cargar modelo
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# -------- FUNCION PARA PROCESAR IMAGEN --------
def procesar_imagen(image):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        temp_path = tmp.name

    results = model(temp_path)
    res_plotted = results[0].plot()

    st.image(res_plotted, caption="Resultado", use_column_width=True)

    st.subheader("📊 Detecciones:")
    boxes = results[0].boxes
    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]
            st.write(f"**{label}** - Confianza: {conf:.2f}")
    else:
        st.write("No se detectaron objetos.")

    os.remove(temp_path)

# -------- OPCIONES DE ENTRADA --------
opcion = st.radio(
    "Selecciona cómo quieres ingresar la imagen:",
    ("📁 Subir imagen", "🌐 URL", "📷 Cámara")
)

# -------- 1. SUBIR IMAGEN --------
if opcion == "📁 Subir imagen":
    uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen cargada", use_column_width=True)

        if st.button("🔍 Detectar"):
            procesar_imagen(image)

# -------- 2. URL --------
elif opcion == "🌐 URL":
    url = st.text_input("Pega el link de la imagen")

    if url:
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content))
            st.image(image, caption="Imagen desde URL", use_column_width=True)

            if st.button("🔍 Detectar"):
                procesar_imagen(image)

        except:
            st.error("No se pudo cargar la imagen desde la URL.")

# -------- 3. CÁMARA --------
elif opcion == "📷 Cámara":
    camera_image = st.camera_input("Toma una foto")

    if camera_image is not None:
        image = Image.open(camera_image)
        st.image(image, caption="Imagen capturada", use_column_width=True)

        if st.button("🔍 Detectar"):
            procesar_imagen(image)
