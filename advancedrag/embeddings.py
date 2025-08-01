import numpy as np
from sentence_transformers import SentenceTransformer
import torch
AVAILABLE_EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": "all-MiniLM-L6-v2",
    "all-MiniLM-L12-v2": "all-MiniLM-L12-v2",
    "all-mpnet-base-v2": "all-mpnet-base-v2",
    "multi-qa-MiniLM-L6-cos-v1": "multi-qa-MiniLM-L6-cos-v1",
    "multi-qa-mpnet-base-dot-v1": "multi-qa-mpnet-base-dot-v1",
    "paraphrase-MiniLM-L6-v2": "paraphrase-MiniLM-L6-v2",
    "paraphrase-mpnet-base-v2": "paraphrase-mpnet-base-v2",
    "distilbert-base-nli-stsb-mean-tokens": "distilbert-base-nli-stsb-mean-tokens"
}

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
embedding_model_name = DEFAULT_EMBEDDING_MODEL
embedding_model = None


def get_embedding_model(model_name=None):
    global embedding_model, embedding_model_name
    if model_name is None:
        model_name = embedding_model_name
    if embedding_model is None or embedding_model_name != model_name:
        try:
            embedding_model = SentenceTransformer(model_name, device='cuda' if torch.cuda.is_available() else 'cpu')
            embedding_model_name = model_name
            print(f"Loaded embedding model: {model_name}")
        except Exception as e:
            print(f"Error loading model {model_name}: {str(e)}")
            if model_name != DEFAULT_EMBEDDING_MODEL:
                embedding_model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL,device='cuda' if torch.cuda.is_available() else 'cpu')
                embedding_model_name = DEFAULT_EMBEDDING_MODEL
                print(f"Fallback to default model: {DEFAULT_EMBEDDING_MODEL}")
    return embedding_model


def set_embedding_model(model_name):
    global embedding_model_name
    embedding_model_name = model_name
    return get_embedding_model(model_name)


def batch_generate_embeddings(texts, batch_size=32, model_name=None):
    current_model = get_embedding_model(model_name)
    num_batches = int(np.ceil(len(texts) / batch_size))
    all_embeddings = []
    for i in range(num_batches):
        start_idx, end_idx = i * batch_size, min((i + 1) * batch_size, len(texts))
        batch = texts[start_idx:end_idx]
        batch_embeddings = current_model.encode(batch)
        all_embeddings.append(batch_embeddings)
    return np.vstack(all_embeddings).astype(np.float32)