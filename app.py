import io

from PIL import Image
import streamlit as st

from models import load_model, translate


def load_image():
    uploaded_file = st.file_uploader(label='Выберите изображение для распознавания')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))


model = load_model()
st.title('Классификация изображений')
img = load_image()
result = st.button('Распознать изображение')
if result:
    st.write('**Результат распознавания:**')
    st.write(translate(*model, img))
