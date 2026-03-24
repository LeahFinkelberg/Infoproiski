# импортирую предобработанный текст и библиотеки
from Text_preprocessing_2 import processed_docs
import gensim
import numpy as np
from numpy.linalg import norm
import re


# задаю функцию, которая пготовит рпедложения для обучения Word2Vec: возвращает список списков токенов и список документов

def prepare_sentences_for_word2vec(processed_docs):
    sentences = []
    doc_names = []

    for doc_id, tokens in processed_docs.items():
        if len(tokens) > 0:
            sentences.append(tokens)
            doc_names.append(doc_id)

    return sentences, doc_names

# обучаю модель на документах, возвращаем обученную модель
def train_word2vec_model(sentences, vector_size=300, window=5, min_count=1, workers=4):

    model = gensim.models.Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=1,
        epochs=10
    )

    return model


# задаю функцию, которая строит векторы документов, усредняя векторы слов
def build_doc_vectors_word2vec(processed_docs, model):
    doc_vectors = {}
    doc_names = list(processed_docs.keys())
    vector_size = model.vector_size

    for doc_id, tokens in processed_docs.items():
        word_vectors = []
        for token in tokens:
            if token in model.wv:
                word_vectors.append(model.wv[token])

        if len(word_vectors) > 0:
            doc_vector = np.mean(word_vectors, axis=0)
        else:
            doc_vector = np.zeros(vector_size)

        doc_vectors[doc_id] = doc_vector

    return doc_vectors, doc_names


# ==================== ЧАСТЬ 3: ПОИСКОВЫЕ ФУНКЦИИ ====================

def cosine_similarity(a, b):
    """Вычисляет косинусную близость между двумя векторами"""
    if norm(a) == 0 or norm(b) == 0:
        return 0.0
    return np.dot(a, b) / (norm(a) * norm(b))


def get_query_vector_word2vec(query, model):
    """
    Получает вектор запроса путем усреднения векторов слов

    Args:
        query: поисковый запрос (строка)
        model: обученная Word2Vec модель

    Returns:
        numpy.ndarray: вектор запроса
    """
    query_tokens = re.findall(r'\w+', query.lower())

    query_vectors = []
    for token in query_tokens:
        if token in model.wv:
            query_vectors.append(model.wv[token])

    if len(query_vectors) > 0:
        return np.mean(query_vectors, axis=0)
    else:
        return np.zeros(model.vector_size)


def search_word2vec(query, model, doc_vectors, doc_names, top_k=5):
    """
    Поиск документов по запросу с использованием Word2Vec

    Args:
        query: поисковый запрос
        model: обученная Word2Vec модель
        doc_vectors: словарь {doc_id: вектор документа}
        doc_names: список названий документов
        top_k: количество возвращаемых результатов

    Returns:
        list of tuples: [(doc_name, score), ...]
    """
    # получаем вектор запроса
    query_vector = get_query_vector_word2vec(query, model)

    # проверяем, не нулевой ли вектор
    if norm(query_vector) == 0:
        return []

    # вычисляем близость со всеми документами
    scores = []
    for doc_id in doc_names:
        doc_vector = doc_vectors[doc_id]
        score = cosine_similarity(query_vector, doc_vector)
        scores.append((doc_id, score))

    # сортируем по убыванию близости
    ranked = sorted(scores, key=lambda x: x[1], reverse=True)

    return ranked[:top_k]
