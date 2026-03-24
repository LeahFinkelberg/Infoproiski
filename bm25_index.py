# импортирую предобработанный текст и библиотеку
from Text_preprocessing_2 import processed_docs
from bm25_vectorizer import BM25Vectorizer

# фунцкия построения обратного индекса через bm25
def build_bm25_index():
# так же, как и с индексом через чатсоты: после предобработки у меня в словаре документам сопоставлены списки токенов, сейчас я из этих списков делаю строки, чтобы с ними мог работать
    docs = [" ".join(tokens) for tokens in processed_docs.values()]
# отдельно создаю список самих документов
    doc_names = list(processed_docs.keys())
# создаю векторизатор
    vectorizer = BM25Vectorizer()
# создаю матрицу, где строки документы, столбцы -- слова, значения -- чатота слова в документе
    X = vectorizer.fit_transform(docs)
# получаю список всех слов
    feature_names = vectorizer.get_feature_names_out()
    bm25_index = {}
# строю обратный индекс; как и в функции для идекса через частоты, матрицу делаю не разреженной, потому что с разреженной код грузился больше 10 минут
    coo = X.tocoo()
    for doc_id, word_id, score in zip(coo.row, coo.col, coo.data):
        word = feature_names[word_id]
        doc = doc_names[doc_id]
        if word not in bm25_index:
            bm25_index[word] = {}
        bm25_index[word][doc] = {}
        bm25_index[word][doc] = float(score)
# возращаю словарь, где ключ -- слово, у него в значении словарь, внуттри которого ключ-- документ, и у него в значении частота слова в докменте
    return bm25_index


# фкнция поиска такая же, как функция поиска для обратного индекса через частоты
def search_bm25(query, top_k=5):
    bm25_index = build_bm25_index()
    query_tokens = query.lower().split()
    scores = {}
    for word in query_tokens:
        if word in bm25_index:
            for doc, score in bm25_index[word].items():
                if doc not in scores:
                    scores[doc] = 0
                scores[doc] += score

    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return ranked_docs[:top_k]
