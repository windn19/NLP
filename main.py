from transformers import pipeline

image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
translate = pipeline('translation_en_to_ru', model='Helsinki-NLP/opus-mt-en-ru')

texts = image_to_text("andrea-rodriguez-YsGtJLKZVgk-unsplash.jpg")
print(translate(texts['generated_text']))
