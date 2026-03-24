# импортирую библиотеки, скачиваю стоп-слова
import nltk
nltk.download('stopwords')
import os
import re
from nltk.corpus import stopwords
import pymorphy3


# создаю морфологический анализтор, беру русские стоп-слова
morph = pymorphy3.MorphAnalyzer()
stop_words = set(stopwords.words("russian"))

# задаю функцию предобработки
def preprocess_text(text):
# привожу к нижнему регистру
    text = text.lower()
# убираю пунктуацию и цифры
    text = re.sub(r'[^а-яё\s]', ' ', text)
# получаю токены
    words = text.split()
# привожу слова к леммам, перед этим удаляю слово, если оно есть в списке стоп-слов
    processed_words = []
    for word in words:
        if word not in stop_words:
            lemma = morph.parse(word)[0].normal_form
            processed_words.append(lemma)
# возвращаю список лемм
    return processed_words



folder_path = r"C:\Users\Lea\PycharmProjects\infopoisk_dz1"
processed_docs = {}

# прохожусь по документам-абзацам исходного текста и прогоняю их через функцию предобработки
for filename in os.listdir(folder_path):
    if filename.startswith("paragraph"):
        path = os.path.join(folder_path, filename)

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        tokens = preprocess_text(text)
        processed_docs[filename] = tokens
