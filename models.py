from transformers import pipeline
import streamlit as st


@st.cache_resource
def load_model():
    image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
    # translate = pipeline('translation_en_to_ru', model='Helsinki-NLP/opus-mt-en-ru')
    return image_to_text # , translate


def translate(img2text, img):
    texts = img2text(img)[0]
    # return trans(texts['generated_text'])
    return texts['generated_text']
