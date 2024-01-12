from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_read_main():
    response = client.get('/')
    assert response.status_code == 200, "Неправильный статус"
    assert response.json() == {'message': 'Hello World!!!'}


def test_translate_right():
    response = client.post('/predict/', json={'text': 'I like Python'})
    json_data = response.json()
    assert response.status_code == 200, 'Неверный статус'
    assert json_data['translation_text'] == 'Мне нравится Питон.'


def test_translate_wrong():
    response = client.post('/predict/', json={'text': 'I like Python'})
    json_data = response.json()
    assert response.status_code == 200, 'Неверный статус'
    assert json_data['translation_text'] != 'Мне нравится мир.'