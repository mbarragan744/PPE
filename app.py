import streamlit as st
from PIL import Image
import torch
from ultralytics import YOLO
import tempfile
import os

# Configuración de la app
st.set_page_config(page_title="Detector con YOLO", layout="centered")
st.title("🖼️ Detección de objetos con YOLO (best.pt)")

# Cargar modelo (solo una vez)
@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    return model

model = load_model()

# Subir imagen
uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mostrar imagen original
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", use_column_width=True)

    # Guardar temporalmente la imagen
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        temp_path = tmp.name

    # Botón para predecir
    if st.button("🔍 Detectar"):
        with st.spinner("Procesando..."):
            results = model(temp_path)

            # Mostrar resultados
            res_plotted = results[0].plot()
            st.image(res_plotted, caption="Resultado", use_column_width=True)

            # Mostrar detecciones en texto
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

    # Limpiar archivo temporal
    os.remove(temp_path)