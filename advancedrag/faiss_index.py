import traceback

import nltk
import os
import tempfile
import shutil
import faiss
import numpy as np
import json

from embeddings import embedding_model_name

def get_faiss_file_paths(chunking_method="standard", model_name=None, base_dir="faiss_files"):
    if model_name is None:
        model_name = embedding_model_name
    model_suffix = model_name.replace("/", "_").replace("-", "_")
    index_dir = os.path.join(base_dir, "index")
    npy_dir = os.path.join(base_dir, "npy")
    json_dir = os.path.join(base_dir, "json")
    for directory in [index_dir, npy_dir, json_dir]:
        os.makedirs(directory, exist_ok=True)
    prefix = "semantic" if chunking_method == "semantic" else "standard"
    index_path = os.path.join(index_dir, f"faiss_index_{prefix}_{model_suffix}.index")
    embeddings_path = os.path.join(npy_dir, f"embeddings_{prefix}_{model_suffix}.npy")
    texts_path = os.path.join(json_dir, f"texts_{prefix}_{model_suffix}.json")
    chunk_metadata_path = os.path.join(json_dir, f"chunk_metadata_{prefix}_{model_suffix}.json")
    return index_path, embeddings_path, texts_path, chunk_metadata_path




def create_faiss_index(embeddings, dimension):
    try:
        index = faiss.IndexHNSWFlat(dimension, 32)
        index.add(embeddings)
        print("Great! Using the HNSWFlat index (faster search, advanced technique).")
        return index
    except AttributeError:
        try:
            index = faiss.IndexFlatIP(dimension)
            faiss.normalize_L2(embeddings)
            index.add(embeddings)
            print("Using IndexFlatIP (cosine similarity-based search).")
            return index
        except AttributeError:
            try:
                index = faiss.IndexFlatL2(dimension)
                index.add(embeddings)
                print("Using IndexFlatL2 (L2 distance-based search).")
                return index
            except Exception:
                print("Looks like all advanced methods failed, so we're falling back on the most basic flat index.")
                index = faiss.IndexFlat(dimension)
                index.add(embeddings)
                return index
    except Exception as e:
        print(f"Oops, something went wrong with the advanced indices: {e}. Falling back to the basic IndexFlat.")
        index = faiss.IndexFlat(dimension)
        index.add(embeddings)
        return index



def save_faiss_data(index, embeddings, texts, metadata=None, chunking_method="standard", model_name=None, base_dir="faiss_files"):
    index_path, embeddings_path, texts_path, chunk_metadata_path = get_faiss_file_paths(chunking_method, model_name, base_dir=base_dir)
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            temp_index_path = os.path.join(temp_dir, "faiss_index.index")
            temp_embeddings_path = os.path.join(temp_dir, "embeddings.npy")
            temp_texts_path = os.path.join(temp_dir, "texts.json")
            temp_metadata_path = os.path.join(temp_dir, "chunk_metadata.json")
            faiss.write_index(index, temp_index_path)
            np.save(temp_embeddings_path, embeddings)
            with open(temp_texts_path, 'w', encoding='utf-8') as f:
                json.dump(texts, f, ensure_ascii=False, indent=2)
            if metadata:
                with open(temp_metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
            os.makedirs(os.path.dirname(texts_path), exist_ok=True)
            if metadata:
                os.makedirs(os.path.dirname(chunk_metadata_path), exist_ok=True)
            shutil.copy2(temp_index_path, index_path)
            shutil.copy2(temp_embeddings_path, embeddings_path)
            shutil.copy2(temp_texts_path, texts_path)
            if metadata:
                shutil.copy2(temp_metadata_path, chunk_metadata_path)
            print(f"FAISS data saved successfully for '{chunking_method}' method with model '{model_name or embedding_model_name}'")
        except Exception as e:
            print(f" Error saving FAISS data: {str(e)}")
            print(traceback.format_exc())
            raise

def load_faiss_data(chunking_method="standard", model_name=None):
    index_path, embeddings_path, texts_path, chunk_metadata_path = get_faiss_file_paths(chunking_method, model_name)
    if not all(os.path.exists(p) for p in [index_path, embeddings_path, texts_path]):
        return None, None, None, None
    try:
        index = faiss.read_index(index_path)
        embeddings = np.load(embeddings_path, mmap_mode='r')
        with open(texts_path, 'r', encoding='utf-8') as f:
            texts = json.load(f)
        metadata = None
        if os.path.exists(chunk_metadata_path):
            with open(chunk_metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        return index, embeddings, texts, metadata
    except Exception as e:
        print(f" Error loading FAISS data: {str(e)}")
        return None, None, None, None