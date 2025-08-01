import faiss
import numpy as np
import re
import spacy

from embeddings import get_embedding_model


nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "tagger"])


def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.text for token in doc if token.is_alpha and not token.is_stop]
    return tokens

def semantic_search(index, texts, metadata, query, n_results=3, model_name=None):
    current_model = get_embedding_model(model_name)
    query_vector = current_model.encode([query]).astype(np.float32)
    faiss.normalize_L2(query_vector)
    extra_results = min(n_results * 3, len(texts))
    distances, indices = index.search(query_vector, extra_results)
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(texts):
            chunk_text = texts[idx]
            if chunk_text and re.match(r'^[A-Z]', chunk_text):
                item = {
                    "text": chunk_text,
                    "metadata": metadata[idx] if metadata else {"doc_index": -1},
                    "score": 1.0 - float(distances[0][i]),
                    "index": idx
                }
                results.append(item)
            if len(results) >= n_results:
                break
    return results


def bm25_search(bm25_model, tokenized_corpus, texts, metadata, query, n_results=10):
    tokenized_query = preprocess_text(query)
    scores = bm25_model.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1]
    results = []
    for idx in top_indices:
        if scores[idx] <= 0:
            continue
        result = {
            "text": texts[idx],
            "metadata": metadata[idx] if metadata else {"doc_index": -1},
            "score": float(scores[idx]),
            "index": int(idx),
            "search_type": "lexical"}
        results.append(result)
        if len(results) >= n_results:
            break
    return results


def hybrid_search(index, bm25_model, tokenized_corpus, texts, metadata, query,n_semantic=7, n_lexical=5, alpha=0.7, n_results=5, model_name=None):
    semantic_results = semantic_search(index, texts, metadata, query, n_semantic, model_name)
    lexical_results = bm25_search(bm25_model, tokenized_corpus, texts, metadata, query, n_lexical)
    combined_results = {}
    if semantic_results:
        semantic_scores = [result["score"] for result in semantic_results]
        min_sem_score = min(semantic_scores)
        max_sem_score = max(semantic_scores)
        # normalizing semantic similarity scores using min-max normalization
        # lower raw distance -> higher score
        # avoid division by 0, make it 1 if identical
        score_range = max_sem_score - min_sem_score if max_sem_score > min_sem_score else 1.0
        for result in semantic_results:
            idx = result["index"]
            normalized_score = 1 - (result["score"] - min_sem_score) / score_range if score_range > 0 else 0.5
            combined_results[idx] = {
                "text": result["text"],
                "metadata": result["metadata"],
                "semantic_score": normalized_score,
                "lexical_score": 0.0,
                "index": idx
            }
    if lexical_results:
        lexical_scores = [result["score"] for result in lexical_results]
        max_lex_score = max(lexical_scores) if lexical_scores else 1.0
        for result in lexical_results:
            idx = result["index"]
            normalized_score = result["score"] / max_lex_score if max_lex_score > 0 else 0.0
            if idx in combined_results:
                combined_results[idx]["lexical_score"] = normalized_score
            else:
                combined_results[idx] = {
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "semantic_score": 0.0,
                    "lexical_score": normalized_score,
                    "index": idx
                }
    for idx in combined_results:
        combined_results[idx]["combined_score"] = (
                alpha * combined_results[idx]["semantic_score"] +
                (1 - alpha) * combined_results[idx]["lexical_score"]
        )
    sorted_results = sorted(
        combined_results.values(),
        key=lambda x: x["combined_score"],
        reverse=True
    )[:n_results]
    final_results = []
    for result in sorted_results:
        final_results.append({
            "text": result["text"],
            "metadata": result["metadata"],
            "score": result["combined_score"],
            "semantic_score": result["semantic_score"],
            "lexical_score": result["lexical_score"],
            "index": result["index"],
            "search_type": "hybrid"
        })
    return final_results

