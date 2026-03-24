# импортируем библиотеки
import fasttext
import numpy as np
from numpy.linalg import norm
# from Text_preprocessing_2 import processed_docs
model_path = "cc.ru.300.bin"


# задаем функцию создания индексов с помощью fasttest: принимает корпусные документы и путь к модели, возвращает модель и список векторов всех документов
def build_fasttext_index(processed_docs, model_path):
# загружаем модель
    model = fasttext.load_model(model_path)

    doc_vectors = {}
# для каждого документа: проходимся по его словам, получаем для каждого слова вектор и усреднем векторы, чтобы получить вектор этого документа
    for doc_id, tokens in processed_docs.items():
        word_vectors = []
        for word in tokens:
            word_vectors.append(model.get_word_vector(word))
        if len(word_vectors) > 0:
            doc_vector = np.mean(word_vectors, axis=0)
        else:
            doc_vector = np.zeros(model.get_dimension())
        doc_vectors[doc_id] = doc_vector

    return model, doc_vectors


# задаем функцию расчета косинусной близости: принимает два вектора, возаращает косинус угла между ними
def cosine_similarity(a, b):
    if norm(a) == 0 or norm(b) == 0:
        return 0
    return np.dot(a, b) / (norm(a) * norm(b))


# задаем функцию поиска: принимает на вход запрос, модель, векторы документов и то, сколько надо в топе; возвращает ранжированный то документов
def search_fasttext(query, model, doc_vectors, top_k=5):

# разделяем запрос по словам и для каждого считаем вектор
    query_tokens = query.lower().split()
    query_vectors = [model.get_word_vector(w) for w in query_tokens]
# считаем средний вектор запроса
    if len(query_vectors) > 0:
        query_vector = np.mean(query_vectors, axis=0)
    else:
        query_vector = np.zeros(model.get_dimension())

    scores = []
# для вектора каждого документа считаем его косинусную близость с вектором запроса
    for doc_id, doc_vector in doc_vectors.items():
        score = cosine_similarity(query_vector, doc_vector)
        scores.append((doc_id, score))
# соритруем по релевантности
    ranked = sorted(scores, key=lambda x: x[1], reverse=True)

    return ranked[:top_k]
