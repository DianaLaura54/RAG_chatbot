import os
import json
import pickle
import traceback

import spacy
from rank_bm25 import BM25Okapi
import tempfile
import shutil
from embeddings import DEFAULT_EMBEDDING_MODEL

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "tagger"])

def get_bm25_file_paths(chunking_method="standard", model_name=None, base_dir="bm25_files"):
    if model_name is None:
        model_name = DEFAULT_EMBEDDING_MODEL
    model_suffix = model_name.replace("/", "_").replace("-", "_")
    pkl_dir = os.path.join(base_dir, "pkl")
    json_dir = os.path.join(base_dir, "json")
    for directory in [pkl_dir, json_dir]:
        os.makedirs(directory, exist_ok=True)
    prefix = "semantic" if chunking_method == "semantic" else "standard"
    bm25_path = os.path.join(pkl_dir, f"bm25_model_{prefix}_{model_suffix}.pkl")
    tokenized_corpus_path = os.path.join(pkl_dir, f"tokenized_corpus_{prefix}_{model_suffix}.pkl")
    texts_path = os.path.join(json_dir, f"texts_{prefix}_{model_suffix}.json")
    chunk_metadata_path = os.path.join(json_dir, f"chunk_metadata_{prefix}_{model_suffix}.json")
    return bm25_path, tokenized_corpus_path, texts_path, chunk_metadata_path


def tokenize_text(text):
    doc = nlp(text.lower())
    tokens = [token.text for token in doc if token.is_alpha and not token.is_stop]
    return tokens


def create_bm25_index(texts):
    tokenized_corpus = [tokenize_text(doc) for doc in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, tokenized_corpus


def save_data(bm25_model, tokenized_corpus, texts, metadata=None, chunking_method="standard", model_name=None, base_dir="bm25_files"):
    bm25_path, tokenized_corpus_path, texts_path, chunk_metadata_path = get_bm25_file_paths(
        chunking_method, model_name, base_dir=base_dir)
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            temp_bm25_path = os.path.join(temp_dir, "bm25_model.pkl")
            temp_tokenized_corpus_path = os.path.join(temp_dir, "tokenized_corpus.pkl")
            temp_texts_path = os.path.join(temp_dir, "texts.json")
            temp_metadata_path = os.path.join(temp_dir, "chunk_metadata.json")
            with open(temp_bm25_path, 'wb') as f:
                pickle.dump(bm25_model, f)
            with open(temp_tokenized_corpus_path, 'wb') as f:
                pickle.dump(tokenized_corpus, f)
            with open(temp_texts_path, 'w', encoding='utf-8') as f:
                json.dump(texts, f, ensure_ascii=False, indent=2)
            if metadata:
                with open(temp_metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
            os.makedirs(os.path.dirname(bm25_path), exist_ok=True)
            os.makedirs(os.path.dirname(tokenized_corpus_path), exist_ok=True)
            os.makedirs(os.path.dirname(texts_path), exist_ok=True)
            if metadata:
                os.makedirs(os.path.dirname(chunk_metadata_path), exist_ok=True)
            shutil.copy2(temp_bm25_path, bm25_path)
            shutil.copy2(temp_tokenized_corpus_path, tokenized_corpus_path)
            shutil.copy2(temp_texts_path, texts_path)
            if metadata:
                shutil.copy2(temp_metadata_path, chunk_metadata_path)
            print(f" BM25 data saved successfully for '{chunking_method}' method with model '{model_name or DEFAULT_EMBEDDING_MODEL}'")
        except Exception as e:
            print(f" Error saving BM25 data: {str(e)}")
            print(traceback.format_exc())
            raise


def load_data(chunking_method="standard", model_name=None):
    bm25_path, tokenized_corpus_path, texts_path, chunk_metadata_path = get_bm25_file_paths(chunking_method, model_name)
    if not all(os.path.exists(p) for p in [bm25_path, tokenized_corpus_path, texts_path]):
        return None, None, None, None
    try:
        with open(bm25_path, 'rb') as f:
            bm25_model = pickle.load(f)
        with open(tokenized_corpus_path, 'rb') as f:
            tokenized_corpus = pickle.load(f)
        with open(texts_path, 'r', encoding='utf-8') as f:
            texts = json.load(f)
        metadata = None
        if os.path.exists(chunk_metadata_path):
            with open(chunk_metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        return bm25_model, tokenized_corpus, texts, metadata
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None, None, None, None

