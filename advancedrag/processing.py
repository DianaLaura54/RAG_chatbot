import os

import PyPDF2
import docx
import pandas as pd
from chunking import chunk_documents_semantic, chunk_documents
from embeddings import embedding_model_name, DEFAULT_EMBEDDING_MODEL, set_embedding_model, get_embedding_model, \
    batch_generate_embeddings
from lexical import create_bm25_index, save_data
from faiss_index import create_faiss_index, save_faiss_data



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


def process_files(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return [(f.read(), 1)]
        elif ext == '.pdf':
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                return [(page.extract_text(), i + 1) for i, page in enumerate(pdf_reader.pages) if page.extract_text()]
        elif ext == '.docx':
            doc = docx.Document(file_path)
            return [(' '.join([p.text for p in doc.paragraphs]), 1)]
        elif ext == '.csv':
            df = pd.read_csv(file_path)
            text_columns = [col for col in df.columns if df[col].dtype == 'O']
            return [(df[text_columns].astype(str).agg(' '.join, axis=1).str.cat(sep=' '), 1)]
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return []


def get_all_files_in_folder(folder_path):
    all_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.txt', '.pdf', '.docx', '.csv')):
                all_files.append(os.path.join(root, file))
    return all_files


def process_documents_from_folder_hybrid(folder_path, chunking_method="standard",min_chunk_size=50, max_chunk_size=250,similarity_threshold=0.6, model_name=None):
    print(f"Processing documents from {folder_path} using {chunking_method} chunking...")
    from chunking import chunk_documents_semantic, chunk_documents
    if model_name:
        set_embedding_model(model_name)
    all_files = get_all_files_in_folder(folder_path)
    if not all_files:
        print(f"No files found in folder: {folder_path}")
        return None, None, None, None, None, None, []
    all_documents_with_pages, file_sources = [], []
    for file_path in all_files:
        file_docs = process_files(file_path)
        if file_docs:
            all_documents_with_pages.append(file_docs)
            file_sources.append(file_path)
    if not all_documents_with_pages:
        print("No document content extracted from files.")
        return None, None, None, None, None, None, []
    if chunking_method == "semantic":
        chunks, metadata = chunk_documents_semantic(
            all_documents_with_pages,
            file_sources,
            get_embedding_model()
        )
    else:
        chunks, metadata = chunk_documents(all_documents_with_pages, file_sources)
    if not chunks:
        print("No chunks created.")
        return None, None, None, None, None, None, []
    print(f"Created {len(chunks)} chunks using {chunking_method} chunking method.")
    print("Generating embeddings...")
    embeddings = batch_generate_embeddings(chunks, model_name=model_name)
    print("Creating FAISS index...")
    index = create_faiss_index(embeddings, embeddings.shape[1])
    print("Creating BM25 index...")
    bm25_model, tokenized_corpus = create_bm25_index(chunks)
    print("Saving data to disk...")
    save_faiss_data(index, embeddings, chunks, metadata, chunking_method, model_name)
    save_data(bm25_model, tokenized_corpus, chunks, metadata, chunking_method, model_name)
    print(f"Successfully saved {chunking_method} chunking data with model {model_name or embedding_model_name}.")
    flat_documents = [doc for doc_list in all_documents_with_pages for doc in doc_list]
    return index, embeddings, chunks, metadata, bm25_model, tokenized_corpus, flat_documents


def process_documents_from_folder_semantic(folder_path, chunking_method="standard", model_name=None,min_chunk_size=50, max_chunk_size=250, similarity_threshold=0.5):
    print(f"Processing documents from {folder_path} using {chunking_method} chunking...")
    from chunking import chunk_documents_semantic, chunk_documents
    if model_name:
        set_embedding_model(model_name)
        print(f"Using embedding model: {model_name}")
    all_files = get_all_files_in_folder(folder_path)
    if not all_files:
        print(f"No files found in folder: {folder_path}")
        return None, None, None, None, []
    all_documents_with_pages, file_sources = [], []
    for file_path in all_files:
        file_docs = process_files(file_path)
        if file_docs:
            all_documents_with_pages.append(file_docs)
            file_sources.append(file_path)
    if not all_documents_with_pages:
        print("No document content extracted from files.")
        return None, None, None, None, []
    if chunking_method == "semantic":
        chunks, metadata = chunk_documents_semantic(all_documents_with_pages,file_sources,embedding_model=get_embedding_model())
    else:
        chunks, metadata = chunk_documents(all_documents_with_pages, file_sources)
    if not chunks:
        print("No chunks created.")
        return None, None, None, None, []
    print(f"Created {len(chunks)} chunks using {chunking_method} method.")
    embeddings = batch_generate_embeddings(chunks, model_name=model_name)
    index = create_faiss_index(embeddings, embeddings.shape[1])
    save_faiss_data(index, embeddings, chunks, metadata, chunking_method, model_name)
    flat_documents = [doc for doc_list in all_documents_with_pages for doc in doc_list]
    return index, embeddings, chunks, metadata, flat_documents


def process_documents_from_folder_lexical(folder_path, chunking_method="standard", model_name=None,min_chunk_size=50, max_chunk_size=250, similarity_threshold=0.5):
    print(f"Processing documents from {folder_path} using {chunking_method} chunking...")
    if model_name:
        set_embedding_model(model_name)
        print(f"Using embedding model: {model_name}")
    all_files = get_all_files_in_folder(folder_path)
    if not all_files:
        print(f"No files found in folder: {folder_path}")
        return None, None, None, None
    all_documents_with_pages, file_sources = [], []
    for file_path in all_files:
        file_docs = process_files(file_path)
        if file_docs:
            all_documents_with_pages.append(file_docs)
            file_sources.append(file_path)
    if not all_documents_with_pages:
        print("No document content extracted from files.")
        return None, None, None, None
    if chunking_method == "semantic":
        chunks, metadata = chunk_documents_semantic(all_documents_with_pages,file_sources,embedding_model=get_embedding_model() if model_name else None)
    else:
        chunks, metadata = chunk_documents(all_documents_with_pages, file_sources)
    if not chunks:
        print("No chunks created.")
        return None, None, None, None
    print(f"Created {len(chunks)} chunks using {chunking_method} method.")
    bm25_model, tokenized_corpus = create_bm25_index(chunks)
    save_data(bm25_model, tokenized_corpus, chunks, metadata, chunking_method, model_name)
    return bm25_model, tokenized_corpus, chunks, metadata