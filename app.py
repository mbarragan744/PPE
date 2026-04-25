import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import tempfile
import os
from ultralytics import YOLO

# Configuración
st.set_page_config(page_title="Detector PPE", layout="centered")
st.title("🦺 Evaluación de Uso de PPE")

# Cargar modelo
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# -------- FUNCIÓN PARA EVALUAR PPE --------
def evaluar_ppe(detecciones, nombres_clases):
    etiquetas = [nombres_clases[int(box.cls[0])] for box in detecciones]

    tiene_casco = "helmet" in etiquetas
    tiene_chaleco = "vest" in etiquetas
    persona = "person" in etiquetas

    if persona:
        if tiene_casco and tiene_chaleco:
            return "✅ PPE COMPLETO", "green"
        elif tiene_casco or tiene_chaleco:
            return "⚠️ PPE INCOMPLETO", "orange"
        else:
            return "❌ SIN PPE", "red"
    else:
        return "ℹ️ No se detecta persona", "blue"

# -------- PROCESAMIENTO --------
def procesar_imagen(image):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        temp_path = tmp.name

    results = model(temp_path)
    res_plotted = results[0].plot()

    st.image(res_plotted, caption="Resultado", use_column_width=True)

    boxes = results[0].boxes
    nombres_clases = model.names

    # Mostrar detecciones
    st.subheader("📊 Detecciones:")
    etiquetas = []

    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = nombres_clases[cls_id]
            etiquetas.append(label)
            st.write(f"**{label}** - Confianza: {conf:.2f}")
    else:
        st.write("No se detectaron objetos.")

    # -------- RESULTADO FINAL PPE --------
    resultado, color = evaluar_ppe(boxes if boxes else [], nombres_clases)

    st.markdown("## 🧾 Resultado final:")
    st.markdown(f"<h2 style='color:{color}'>{resultado}</h2>", unsafe_allow_html=True)

    os.remove(temp_path)

# -------- OPCIONES --------
opcion = st.radio(
    "Selecciona cómo ingresar la imagen:",
    ("📁 Subir imagen", "🌐 URL", "📷 Cámara")
)

# -------- SUBIR --------
if opcion == "📁 Subir imagen":
    uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen cargada", use_column_width=True)

        if st.button("🔍 Detectar"):
            procesar_imagen(image)

# -------- URL --------
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
            st.error("No se pudo cargar la imagen.")

# -------- CÁMARA --------
elif opcion == "📷 Cámara":
    camera_image = st.camera_input("Toma una foto")

    if camera_image is not None:
        image = Image.open(camera_image)
        st.image(image, caption="Imagen capturada", use_column_width=True)

        if st.button("🔍 Detectar"):
            procesar_imagen(image)

        if st.button("🔍 Detectar"):
            procesar_imagen(image)
