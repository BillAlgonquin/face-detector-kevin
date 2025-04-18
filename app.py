import streamlit as st
from PIL import Image
import torch

st.title("Detección de Rostro - Kevin")
st.write("Sube una imagen para detectar tu rostro usando el modelo personalizado.")

model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', source='local')

uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen original", use_column_width=True)

    results = model(image)
    results.render()
    st.image(results.ims[0], caption="Detección", use_column_width=True)
    st.write(results.pandas().xyxy[0])
