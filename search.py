import argparse
import time
import sys

from Text_preprocessing_2 import processed_docs
from bm25_index import search_bm25
from word_to_vec_index import (
    prepare_sentences_for_word2vec,
    train_word2vec_model,
    build_doc_vectors_word2vec,
    search_word2vec
)
from fasttext_index import build_fasttext_index, search_fasttext


# кеш
_word2vec_model = None
_word2vec_doc_vectors = None
_word2vec_doc_names = None


def load_or_train_word2vec(processed_docs, vector_size=300, window=5, min_count=1, workers=4):
    global _word2vec_model, _word2vec_doc_vectors, _word2vec_doc_names

    if _word2vec_model is None:
        sentences, doc_names = prepare_sentences_for_word2vec(processed_docs)

        _word2vec_model = train_word2vec_model(
            sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers
        )

        _word2vec_doc_vectors, _ = build_doc_vectors_word2vec(processed_docs, _word2vec_model)
        _word2vec_doc_names = doc_names

    return _word2vec_model, _word2vec_doc_vectors, _word2vec_doc_names


def main():
    parser = argparse.ArgumentParser(description="Search Engine")

    parser.add_argument('-q', '--query', type=str, help='Search query')
    parser.add_argument('-m', '--model', type=str,
                        choices=['bm25', 'word2vec', 'fasttext'],
                        default='bm25')
    parser.add_argument('-k', '--top-k', type=int, default=5)
    parser.add_argument('-i', '--interactive', action='store_true')
    parser.add_argument('--model-path', type=str, default='cc.ru.300.bin')

    args = parser.parse_args()

    if args.interactive:
        print("Interactive mode (bm25 / word2vec / fasttext)")
        current_model = args.model

        fasttext_model = None
        fasttext_vectors = None
        word2vec_model = None
        word2vec_vectors = None
        word2vec_names = None

        while True:
            query = input(f"\n[{current_model}] Search: ").strip()

            if query in ['exit', 'quit']:
                break

            if query.startswith("model "):
                current_model = query.split()[1]
                print(f"Switched to {current_model}")
                continue

            try:
                if current_model == 'bm25':
                    results = search_bm25(query, top_k=args.top_k)

                elif current_model == 'word2vec':
                    if word2vec_model is None:
                        word2vec_model, word2vec_vectors, word2vec_names = load_or_train_word2vec(processed_docs)

                    results = search_word2vec(
                        query,
                        word2vec_model,
                        word2vec_vectors,
                        word2vec_names,
                        top_k=args.top_k
                    )

                elif current_model == 'fasttext':
                    if fasttext_model is None:
                        fasttext_model, fasttext_vectors = build_fasttext_index(processed_docs, args.model_path)

                    results = search_fasttext(query, fasttext_model, fasttext_vectors, top_k=args.top_k)

                else:
                    print("Unknown model")
                    continue

                for i, (doc, score) in enumerate(results, 1):
                    print(f"{i}. {doc} ({score:.4f})")

            except Exception as e:
                print(f"Error: {e}")

        return

    # обычный режим
    if not args.query:
        print("Provide query with -q")
        return

    start = time.time()

    try:
        if args.model == 'bm25':
            results = search_bm25(args.query, top_k=args.top_k)

        elif args.model == 'word2vec':
            model, vecs, names = load_or_train_word2vec(processed_docs)
            results = search_word2vec(args.query, model, vecs, names, top_k=args.top_k)

        elif args.model == 'fasttext':
            model, vecs = build_fasttext_index(processed_docs, args.model_path)
            results = search_fasttext(args.query, model, vecs, top_k=args.top_k)

        print(f"\nTime: {time.time() - start:.4f}s")

        for i, (doc, score) in enumerate(results, 1):
            print(f"{i}. {doc} ({score:.4f})")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
