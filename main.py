from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()
translate = pipeline('translation_en_to_ru', model='Helsinki-NLP/opus-mt-en-ru')


class Item(BaseModel):
    text: str


@app.get('/')
def root():
    return {'message': 'Hello World!!!'}


@app.post('/predict/')
def predict(item: Item):
    """Перевод текста"""
    return translate(item.text)
