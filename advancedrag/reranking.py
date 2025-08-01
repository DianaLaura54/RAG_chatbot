from sentence_transformers import CrossEncoder


AVAILABLE_RERANKER_MODELS = {
    "cross-encoder/stsb-roberta-base",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "cross-encoder/ms-marco-TinyBERT-L-2-v2",
    "cross-encoder/ms-marco-electra-base",
    "cross-encoder/qnli-electra-base"
    }



def get_available_reranker_models():
    return AVAILABLE_RERANKER_MODELS


def get_reranker_model_names():
    return list(AVAILABLE_RERANKER_MODELS.keys())


def get_default_reranker_model():
    return "cross-encoder/stsb-roberta-base"


def reranker(query, hits, model_name="cross-encoder/stsb-roberta-base"):
    #predict similarity scores between the query and each retrieved hit
    if not hits:
        return hits
    try:
        cross_encoder_model = CrossEncoder(model_name)
        sentence_pairs = [[query, hit["text"]] for hit in hits]
        similarity_scores = cross_encoder_model.predict(sentence_pairs)
        for idx in range(len(hits)):
            hits[idx]["cross-encoder_score"] = float(similarity_scores[idx])
        hits = sorted(hits, key=lambda x: x["cross-encoder_score"], reverse=True)
        return hits
    except Exception as e:
        print(f"Error in reranker with model {model_name}: {str(e)}")
        for hit in hits:
            hit["cross-encoder_score"] = 0.0
        return hits


def rerank_search_results(search_results, query, model_name="cross-encoder/stsb-roberta-base"):
    reranked_results = {}
    for method, hits in search_results.items():
        if hits:
            reranked_results[method] = reranker(query, hits, model_name)
        else:
            reranked_results[method] = []
    return reranked_results