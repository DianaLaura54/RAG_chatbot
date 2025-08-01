import time

import streamlit as st
import pandas as pd
import os
import re

from chunking import chunk_documents, chunk_documents_semantic
from embeddings import AVAILABLE_EMBEDDING_MODELS, set_embedding_model, batch_generate_embeddings
from generation import query_llama3, query_mistral
from processing import get_all_files_in_folder, process_files, get_faiss_file_paths, get_bm25_file_paths
from search import hybrid_search, bm25_search, semantic_search
from styling.styles import get_css
from faiss_index import (
    load_faiss_data,
    create_faiss_index,
    save_faiss_data,
)
from common import (
    log_max_bertscore_to_csv,
    get_random_question,
    format_source_info,
    is_valid_input,
    check_images_extracted,
    optimize_query
)
from evaluate import (
    compute_bert_score,
    compute_bertscore_with_filter,
    filter_chunks_by_pdf,
    compute_rougel_with_filter, compute_rouge_l_score,
)

from lexical import (
    create_bm25_index,
    save_data,
    load_data
)

from reranking import reranker, get_available_reranker_models, get_default_reranker_model
from extract import (extract_images_from_pdf, process_pdf_folder)

from audio import generate_audio_from_text, clean_text_for_audio

bm25_path = "bm25_model.pkl"
tokenized_corpus_path = "tokenized_corpus.pkl"
y = "No"


def perform_multi_method_search(user_input, search_data, num_results=5, selected_pdf="All PDFs", use_reranker=True):
    chunking_method = search_data.get('current_chunking_method', 'standard')
    faiss_index = search_data['faiss_index']
    texts = search_data['texts']
    metadata = search_data['metadata']
    bm25_model = search_data['bm25_model']
    tokenized_corpus = search_data['tokenized_corpus']
    model_name = search_data.get('embedding_model_name')
    initial_num_results = num_results * 3 if use_reranker else num_results
    semantic_results = semantic_search(
        faiss_index,
        texts,
        metadata,
        user_input,
        n_results=initial_num_results,
        model_name=model_name
    )
    for result in semantic_results:
        result['chunking_method'] = chunking_method
    lexical_results = bm25_search(
        bm25_model,
        tokenized_corpus,
        texts,
        metadata,
        user_input,
        n_results=initial_num_results
    )
    for result in lexical_results:
        result['chunking_method'] = chunking_method
    alpha = st.session_state.get('hybrid_alpha', 0.7)
    n_semantic = st.session_state.get('n_semantic', 7)
    n_lexical = st.session_state.get('n_lexical', 5)
    if chunking_method == "semantic":
        adjusted_alpha = min(alpha + 0.1, 0.9)
    else:
        adjusted_alpha = alpha
    hybrid_results = hybrid_search(
        faiss_index,
        bm25_model,
        tokenized_corpus,
        texts,
        metadata,
        user_input,
        n_semantic=n_semantic,
        n_lexical=n_lexical,
        alpha=adjusted_alpha,
        n_results=initial_num_results,
        model_name=model_name
    )
    for result in hybrid_results:
        result['chunking_method'] = chunking_method
        result['adjusted_alpha'] = adjusted_alpha
    if selected_pdf != "All PDFs":
        semantic_filtered = filter_chunks_by_pdf(semantic_results, selected_pdf)
        lexical_filtered = filter_chunks_by_pdf(lexical_results, selected_pdf)
        hybrid_filtered = filter_chunks_by_pdf(hybrid_results, selected_pdf)
        semantic_results = semantic_filtered if semantic_filtered else semantic_results
        lexical_results = lexical_filtered if lexical_filtered else lexical_results
        hybrid_results = hybrid_filtered if hybrid_filtered else hybrid_results
    search_results = {
        "semantic": semantic_results,
        "lexical": lexical_results,
        "hybrid": hybrid_results
    }
    # check if reranker should be used
    if use_reranker:
        # get reranker model from session state or use default
        reranker_model_name = st.session_state.get("reranker_model", get_default_reranker_model())
        reranked_results = {}
        # iterate over each search method and rerank results if available
        for method, hits in search_results.items():
            if hits:
                # apply reranker and limit to num_results
                reranked_results[method] = reranker(user_input, hits, reranker_model_name)[:num_results]
            else:
                # if no hits, set empty list
                reranked_results[method] = []
        # return reranked results
        return reranked_results
    else:
        # limit each method's results to num_results without reranking
        for method in search_results:
            search_results[method] = search_results[method][:num_results]
        # return original search results truncated
        return search_results

def process_pdf_images_wrapper(folder_path, output_base_folder="extracted_images", recursive=True):
    if check_images_extracted(folder_path, output_base_folder):
        total_pdfs = 0
        total_images = 0
        for pdf in [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]:
            pdf_name = os.path.splitext(pdf)[0]
            pdf_images_folder = os.path.join(output_base_folder, pdf_name)
            if os.path.exists(pdf_images_folder):
                total_pdfs += 1
                image_files = [f for f in os.listdir(pdf_images_folder)
                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                total_images += len(image_files)

        return {}, total_images, total_pdfs
    all_results = {}
    total_images = 0
    total_pdfs = 0
    if os.path.isfile(folder_path) and folder_path.lower().endswith('.pdf'):
        pdf_path = folder_path
        pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
        pdf_output_folder = os.path.join(output_base_folder, pdf_filename)
        os.makedirs(pdf_output_folder, exist_ok=True)
        try:
            extracted_images = extract_images_from_pdf(pdf_path, pdf_output_folder)
            all_results[pdf_path] = extracted_images
            return all_results, len(extracted_images), 1
        except Exception as e:
            st.error(f"Error processing {pdf_path}: {str(e)}")
            all_results[pdf_path] = []
            return all_results, 0, 1
    all_results = process_pdf_folder(folder_path, output_base_folder, recursive)
    total_pdfs = len(all_results)
    total_images = sum(len(images) for images in all_results.values())
    return all_results, total_images, total_pdfs


def generate_multi_method_response(user_input, search_results, llm_model="llama3"):
    if llm_model == "mistral":
        query_function = query_mistral
    else:
        query_function = query_llama3
    prompt_template = (
        "Hey there! I'll help you find the answer to your question based on these stories:\n"
        "{relevant_documents}\n\n"
        "Here's your question: {user_input}\n"
        "If the answer isn't in the stories, I'll just say 'I don't know'."
    )
    responses = {}
    for method, chunks in search_results.items():
        if not chunks:
            responses[method] = {
                "response": f"No relevant documents found using {method} search.",
                "chunks": [],
                "bert_score": 0.0,
                "rouge_l_score": 0.0
            }
            continue
        response_text = query_function(
            prompt_template,
            user_input,
            chunks
        )
        bert_chunk_scores, chunks_used = compute_bertscore_with_filter(chunks, response_text)
        bert_score = max(bert_chunk_scores) if bert_chunk_scores else 0.0
        rouge_chunk_scores, _ = compute_rougel_with_filter(chunks, response_text)
        rouge_l_score = max(rouge_chunk_scores) if rouge_chunk_scores else 0.0
        if bert_chunk_scores:
            most_relevant_idx = bert_chunk_scores.index(bert_score)
            most_relevant_chunk = chunks_used[most_relevant_idx]
        else:
            most_relevant_chunk = {"text": "No chunk found"}
        source_info = format_source_info(most_relevant_chunk)
        responses[method] = {
            "response": response_text,
            "chunks": chunks,
            "bert_score": bert_score,
            "rouge_l_score": rouge_l_score,
            "source_info": source_info
        }
    return responses


def findanswer(user_question, user_answer=None):
    csv_file_path = os.path.join('Contents',"file.csv")
    try:
        df = pd.read_csv(csv_file_path, encoding='utf-8-sig',sep=';')
        for index, row in df.iterrows():
            if user_question.lower() == row['question'].lower():
                correct_answer = row['answer']
                feedback_msg = ""
                if user_answer:
                    if user_answer.lower() == correct_answer.lower():
                        feedback_msg = "<span style='color:green; font-weight:bold;'>Correct answer!</span>"
                    else:
                        feedback_msg = f"<span style='color:orange; font-weight:bold;'>The correct answer is: {correct_answer}</span>"
                return correct_answer, feedback_msg
        return None, ""
    except FileNotFoundError:
        st.warning(f"Questions file 'file.csv' not found.")
        return None, ""
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")
        return None, ""


def format_multi_method_response(user_input, multi_responses, csv_answer=None, feedback_msg=""):
    chunking_method = st.session_state.get('chunking_method', 'standard')
    chunking_display = "Standard" if chunking_method == "standard" else "Semantic"
    embedding_model = st.session_state.get('embedding_model', 'all-MiniLM-L6-v2')
    response = f"<h3>Results</h3>"
    response += f"<p><small>Using {chunking_display} chunking method with {embedding_model} embeddings</small></p>"
    if csv_answer:
        response += f"<div style='background-color: #f7dae7; padding: 10px; border-radius: 5px; margin-bottom: 15px;'>"
        response += f"<strong>Knowledge Base Answer:</strong> {csv_answer}"
        response += f"{feedback_msg}</div>"
    best_method = None
    best_bert_score = -1
    for method in ["hybrid", "semantic", "lexical"]:
        if method in multi_responses:
            bert_score = multi_responses[method]["bert_score"]
            if bert_score > best_bert_score:
                best_bert_score = bert_score
                best_method = method
    if best_method:
        st.session_state.best_response_for_audio = multi_responses[best_method]["response"]
    for method in ["hybrid", "semantic", "lexical"]:
        if method not in multi_responses:
            continue
        method_data = multi_responses[method]
        method_response = method_data["response"]
        bert_score = method_data["bert_score"]
        rouge_l_score = method_data.get("rouge_l_score", 0.0)
        source_info = method_data.get("source_info", "Unknown source")
        method_info = ""
        if method == "hybrid" and "chunks" in method_data and method_data["chunks"]:
            chunk = method_data["chunks"][0]
            if "adjusted_alpha" in chunk:
                adjusted_alpha = chunk.get("adjusted_alpha", 0.7)
                method_info = f" | Alpha: {adjusted_alpha:.2f}"
        reranker_info = ""
        if "chunks" in method_data and method_data["chunks"] and "cross-encoder_score" in method_data["chunks"][0]:
            top_chunk = method_data["chunks"][0]
            reranker_score = top_chunk.get("cross-encoder_score", 0.0)
        response += f"<div style='background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-bottom: 15px;'>"
        response += f"<h4>{method.capitalize()} Search (BERTScore: {bert_score:.4f} | Rouge-L: {rouge_l_score:.4f}{method_info})</h4>"
        response += f"<p>{method_response}</p>"
        response += f"<p><strong>Source:</strong> {source_info}</p>"
        response += "</div>"
    return response


def scan_manual_folders(images_folder="extracted_images"):
    manual_folders = {}
    if not os.path.exists(images_folder):
        st.sidebar.warning(f"Image folder {images_folder} not found!")
        return manual_folders
    try:
        subfolders = [f for f in os.listdir(images_folder) if os.path.isdir(os.path.join(images_folder, f))]
        for manual_name in subfolders:
            manual_path = os.path.join(images_folder, manual_name)
            image_extensions = ['.jpg', '.jpeg', '.png']
            manual_images = []
            for root, dirs, files in os.walk(manual_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        manual_images.append(os.path.join(root, file))
            page_images = {}
            for img_path in manual_images:
                img_filename = os.path.basename(img_path)
                # search for something like page_123_img
                match = re.search(r'page_(\d+)_img_', img_filename)
                if match:
                    # in this case takes the value 123
                    page_num = int(match.group(1))
                    if page_num not in page_images:
                        page_images[page_num] = []
                    page_images[page_num].append(img_path)
            manual_folders[manual_name] = page_images
        return manual_folders
    except Exception as e:
        st.sidebar.error(f"Error scanning folders: {str(e)}")
        return {}


def log_multi_method_results(question, responses, actual_answer=None, selected_pdf=None, llm_model=None):
    for method, response_data in responses.items():
        response_text = response_data["response"]
        bert_score = response_data["bert_score"]
        rouge_l_score = response_data.get("rouge_l_score", 0.0)
        response_answer_bert_score = None
        max_chunk_answer_bert_score = None
        response_answer_rouge_l = None
        max_chunk_answer_rouge_l = None
        if actual_answer:
            answer_chunk = [{"text": actual_answer}]
            response_answer_bert_score = compute_bert_score(answer_chunk, response_text)
            response_answer_rouge_l = compute_rouge_l_score(answer_chunk, response_text)
            bert_chunk_answer_scores = []
            rouge_chunk_answer_scores = []
            for chunk in response_data["chunks"]:
                chunk_answer_bert_score = compute_bert_score(answer_chunk, chunk["text"])
                chunk_answer_rouge_l = compute_rouge_l_score(answer_chunk, chunk["text"])
                bert_chunk_answer_scores.append(chunk_answer_bert_score)
                rouge_chunk_answer_scores.append(chunk_answer_rouge_l)
            if bert_chunk_answer_scores:
                max_chunk_answer_bert_score = max(bert_chunk_answer_scores)
            if rouge_chunk_answer_scores:
                max_chunk_answer_rouge_l = max(rouge_chunk_answer_scores)
        chunking_method = st.session_state.get('chunking_method', 'standard')
        embedding_model = st.session_state.get('embedding_model', 'all-MiniLM-L6-v2')
        log_max_bertscore_to_csv(
            question,
            response_text,
            actual_answer,
            bert_score,
            response_answer_bert_score,
            max_chunk_answer_bert_score,
            selected_pdf,
            llm_model,
            f"{method}_{chunking_method}",
            rouge_l_score,
            response_answer_rouge_l,
            max_chunk_answer_rouge_l,
            st.session_state.get('use_reranker', True),
            st.session_state.get('reranker_model', get_default_reranker_model()),
            chunking_method,
            st.session_state.get('use_query_optimization'),
            embedding_model
        )


def main():
    base_path = "Contents"
    st.set_page_config(page_title="Chatbot", layout="wide", initial_sidebar_state="collapsed")
    st.markdown(get_css(), unsafe_allow_html=True)
    for key, default_value in {
        'chat_history': [], 'bert_scores': [], 'all_chunk_scores': [],
        'top_chunks': [], 'selected_chunk_index': -1, 'displayed_chunk_scores': False,
        'original_documents': [], 'tts_audio': {}, 'llm_model': "llama3",
        'uploaded_files_info': [], 'chunking_method': 'standard',
        'embedding_model': 'all-MiniLM-L6-v2', 'search_data': None,
        'last_used_chunks': [],
        'rouge_l_scores': [], 'all_rouge_scores': []
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
    if 'indices_initialized' not in st.session_state:
        st.session_state.indices_initialized = False
    if 'initialization_in_progress' not in st.session_state:
        st.session_state.initialization_in_progress = False
    images_output_folder = "extracted_images"
    folder_path = os.path.join(base_path, "books")
    file_csv_path = os.path.join(base_path, "file.csv")
    if not os.path.exists(folder_path):
        st.error(f"Folder not found at: {folder_path}")
        st.info("Please create the folder and add your PDF files.")
        return
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    if not pdf_files:
        st.error(f"No PDF files found in: {folder_path}")
        st.info("Please add PDF files to the folder.")
        return
    with st.sidebar:
        st.subheader("Embedding Model")
        embedding_options = list(AVAILABLE_EMBEDDING_MODELS.keys())
        selected_embedding = st.selectbox(
            "Select Embedding Model",
            options=embedding_options,
            index=embedding_options.index(
                st.session_state.embedding_model) if st.session_state.embedding_model in embedding_options else 0,
            key="embedding_model_selector",
            help="Choose the sentence transformer model for generating embeddings"
        )
        if selected_embedding != st.session_state.embedding_model:
            st.session_state.embedding_model = selected_embedding
            st.session_state.reload_data = True
            st.session_state.indices_initialized = False
            st.session_state.search_data = None
            st.info(f"Embedding model changed to {selected_embedding}. Data will be reloaded.")
        st.subheader("Chunking Method")
        chunking_options = ["Standard", "Semantic"]
        selected_chunking = st.selectbox(
            "Select Chunking Method",
            options=chunking_options,
            index=0 if st.session_state.chunking_method == 'standard' else 1,
            key="chunking_method_selector",
            help="Standard: Fixed-size chunks. Semantic: Content-aware chunking."
        )
        new_chunking_method = "standard" if selected_chunking == "Standard" else "semantic"
        if new_chunking_method != st.session_state.chunking_method:
            st.session_state.chunking_method = new_chunking_method
            st.session_state.reload_data = True
            st.session_state.search_data = None
        st.subheader("Settings")
        if 'pdf_files' in st.session_state and st.session_state.pdf_files:
            pdf_options = ["All PDFs"]
            for pdf_file in st.session_state.pdf_files:
                pdf_options.append(os.path.basename(pdf_file))
            if 'uploaded_files_info' in st.session_state:
                for file_info in st.session_state.uploaded_files_info:
                    if file_info['filename'].lower().endswith('.pdf') and file_info['filename'] not in pdf_options:
                        pdf_options.append(file_info['filename'])
            selected_pdf = st.selectbox(
                "Select a Document",
                options=pdf_options,
                index=0,
                key="pdf_selector"
            )
            st.session_state.selected_pdf = selected_pdf
        st.subheader("Query optimization")
        use_query_optimization = st.checkbox(
            "Use Query Optimization",
            value=False,
            help="Improves search by generating variations of your query"
        )
        st.session_state.use_query_optimization = use_query_optimization
        st.subheader("Search Method")
        search_methods = ["All Methods", "Semantic (FAISS)", "Lexical (BM25)", "Hybrid"]
        selected_search = st.selectbox(
            "Select Search Method",
            options=search_methods,
            index=0,
            key="search_method_selector"
        )
        st.session_state.search_method = selected_search
        st.subheader("Reranker Settings")
        use_reranker = st.checkbox(
            "Use Cross-Encoder Reranker",
            value=True,
            help="Improves ranking quality using a cross-encoder model"
        )
        st.session_state.use_reranker = use_reranker
        if use_reranker:
            reranker_models = list(get_available_reranker_models())
            current_reranker = st.session_state.get('reranker_model', get_default_reranker_model())
            if current_reranker not in reranker_models:
                current_reranker = get_default_reranker_model()
                st.session_state.reranker_model = current_reranker
            reranker_model = st.selectbox(
                "Reranker Model",
                options=reranker_models,
                index=reranker_models.index(current_reranker) if current_reranker in reranker_models else 0,
                help="Choose a reranker model (larger models = better quality but slower)"
            )
            st.session_state.reranker_model = reranker_model
            retrieval_multiplier = st.slider(
                "Retrieval Multiplier",
                min_value=1,
                max_value=5,
                value=3,
                help="How many documents to retrieve initially before reranking (higher = more recall)"
            )
            st.session_state.retrieval_multiplier = retrieval_multiplier
        if selected_search == "Hybrid" or selected_search == "All Methods":
            with st.expander("Hybrid Search Parameters", expanded=False):
                alpha = st.slider("Semantic Weight (α)", min_value=0.0,
                                  max_value=1.0,
                                  value=0.7,
                                  step=0.1,
                                  help="Weight for semantic search (1-α for lexical search)"
                                  )
                st.session_state.hybrid_alpha = alpha
                n_semantic = st.number_input(
                    "Semantic Results",
                    min_value=1,
                    max_value=15,
                    value=7,
                    help="Number of results to retrieve from semantic search"
                )
                st.session_state.n_semantic = n_semantic
                n_lexical = st.number_input(
                    "Lexical Results",
                    min_value=1,
                    max_value=15,
                    value=5,
                    help="Number of results to retrieve from lexical search"
                )
                st.session_state.n_lexical = n_lexical
        st.subheader("LLM Model")
        llm_options = ["llama3", "mistral"]
        selected_llm = st.selectbox(
            "Select LLM Model",
            options=llm_options,
            index=0,
            key="llm_selector"
        )
        st.session_state.llm_model = selected_llm

    def ensure_search_data_loaded():
        try:
            chunking_method = st.session_state.chunking_method
            embedding_model = st.session_state.embedding_model
            faiss_standard_path, _, _, _ = get_faiss_file_paths("standard", embedding_model)
            faiss_semantic_path, _, _, _ = get_faiss_file_paths("semantic", embedding_model)
            bm25_standard_path, _, _, _ = get_bm25_file_paths("standard", embedding_model)
            bm25_semantic_path, _, _, _ = get_bm25_file_paths("semantic", embedding_model)
            standard_exists = os.path.exists(faiss_standard_path) and os.path.exists(bm25_standard_path)
            semantic_exists = os.path.exists(faiss_semantic_path) and os.path.exists(bm25_semantic_path)
            if not standard_exists or not semantic_exists:
                st.info(" First time setup: Creating search indices from your PDF files...")
                set_embedding_model(embedding_model)
                all_files = get_all_files_in_folder(folder_path)
                if not all_files:
                    st.error("No files found in the folder.")
                    return False
                with st.spinner(" Processing PDF documents..."):
                    all_documents_with_pages, file_sources = [], []
                    for file_path in all_files:
                        file_docs = process_files(file_path)
                        if file_docs:
                            all_documents_with_pages.append(file_docs)
                            file_sources.append(file_path)
                if not all_documents_with_pages:
                    st.error("No documents could be processed.")
                    return False
                if not standard_exists:
                    with st.spinner("Creating standard chunks and embeddings..."):
                        chunks, metadata = chunk_documents(all_documents_with_pages, file_sources)
                        embeddings = batch_generate_embeddings(chunks, model_name=embedding_model)
                        index = create_faiss_index(embeddings, embeddings.shape[1])
                        save_faiss_data(index, embeddings, chunks, metadata, chunking_method="standard",model_name=embedding_model)
                        bm25_model, tokenized_corpus = create_bm25_index(chunks)
                        save_data(bm25_model, tokenized_corpus, chunks, metadata, "standard", embedding_model)
                        st.success(" Standard chunking completed!")
                if not semantic_exists:
                    with st.spinner(" Creating semantic chunks and embeddings..."):
                        try:
                            chunks, metadata = chunk_documents_semantic(all_documents_with_pages, file_sources)
                            embeddings = batch_generate_embeddings(chunks, model_name=embedding_model)
                            index = create_faiss_index(embeddings, embeddings.shape[1])
                            save_faiss_data(index, embeddings, chunks, metadata, chunking_method="semantic",
                                            model_name=embedding_model)
                            bm25_model, tokenized_corpus = create_bm25_index(chunks)
                            save_data(bm25_model, tokenized_corpus, chunks, metadata, "semantic", embedding_model)
                            st.success(" Semantic chunking completed!")
                        except Exception as e:
                            st.warning(f"Semantic chunking failed: {str(e)}")
                            st.info("Using standard chunking for semantic method...")
                            chunks, metadata = chunk_documents(all_documents_with_pages, file_sources)
                            embeddings = batch_generate_embeddings(chunks, model_name=embedding_model)
                            index = create_faiss_index(embeddings, embeddings.shape[1])
                            save_faiss_data(index, embeddings, chunks, metadata, chunking_method="semantic", model_name=embedding_model)
                            bm25_model, tokenized_corpus = create_bm25_index(chunks)
                            save_data(bm25_model, tokenized_corpus, chunks, metadata, "semantic", embedding_model)
                st.success(" Search indices created successfully!")
            with st.spinner(f"Loading {chunking_method} chunking data..."):
                set_embedding_model(embedding_model)
                index, embeddings, texts, metadata = load_faiss_data(chunking_method, embedding_model)
                bm25_model, tokenized_corpus, bm25_texts, bm25_metadata = load_data(chunking_method, embedding_model)
                if index is not None and texts and bm25_model and tokenized_corpus:
                    final_texts = texts if bm25_texts is None else bm25_texts
                    final_metadata = metadata if bm25_metadata is None else bm25_metadata
                    st.session_state.search_data = {
                        'faiss_index': index,
                        'texts': final_texts,
                        'metadata': final_metadata,
                        'bm25_model': bm25_model,
                        'tokenized_corpus': tokenized_corpus,
                        'current_chunking_method': chunking_method,
                        'embeddings': embeddings,
                        'embedding_model_name': embedding_model
                    }
                    st.session_state.pdf_files = [f for f in get_all_files_in_folder(folder_path) if f.lower().endswith('.pdf')]
                    if 'selected_pdf' not in st.session_state:
                        st.session_state.selected_pdf = "All PDFs"
                    st.session_state.manual_folders = scan_manual_folders(images_output_folder)
                    images_already_extracted = check_images_extracted(folder_path, images_output_folder)
                    if not images_already_extracted:
                        with st.spinner("Extracting images from PDFs..."):
                            results, total_images, total_pdfs = process_pdf_images_wrapper(folder_path,images_output_folder)
                    return True
                else:
                    st.error("Failed to load search data after creation.")
                    return False
        except Exception as e:
            st.error(f"Error during initialization: {str(e)}")
            return False
    needs_initialization = (
            st.session_state.search_data is None or
            st.session_state.get('reload_data', False) or
            not st.session_state.indices_initialized
    )
    if needs_initialization:
        if ensure_search_data_loaded():
            st.session_state.indices_initialized = True
            st.session_state.reload_data = False
            st.success(" System ready! You can now ask questions about your documents.")
            time.sleep(1)
            st.rerun()
        else:
            st.error("Failed to initialize the search system.")
            st.stop()
    st.markdown(f'<h1 class="main-header">StorySage</h1>', unsafe_allow_html=True)
    top_cols = st.columns([3, 1])
    with top_cols[0]:
        st.markdown('<div class="generate-button" style="margin-bottom: 15px;">', unsafe_allow_html=True)
        if st.button("🎲 Generate Random Question", key="generate_question_btn"):
            random_question = get_random_question()
            st.session_state.user_question_input_chat = random_question
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with top_cols[1]:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Chat", key="clear_chat_btn"):
                for key in ['chat_history', 'bert_scores', 'all_chunk_scores', 'top_chunks', 'selected_chunk_index',
                            'displayed_chunk_scores', 'tts_audio', 'last_used_chunks', 'rouge_l_scores','all_rouge_scores']:
                    if key in st.session_state:
                        if isinstance(st.session_state[key], list):
                            st.session_state[key] = []
                        elif isinstance(st.session_state[key], dict):
                            st.session_state[key] = {}
                        else:
                            st.session_state[key] = -1
                st.rerun()
        with col2:
            if st.button("Metrics", key="metrics_btn"):
                page_name = "pages/metrics.py"
                st.switch_page(page_name)
            if st.button("Go to BERTscore logs", key="BERT_btn"):
                page_name = "pages/knowledge_and_scores_viewer.py"
                st.switch_page(page_name)
            if st.button("Metrics for all", key="metrics2_btn"):
                page_name = "pages/metrics_for_all.py"
                st.switch_page(page_name)
    num_results = 5
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    if not st.session_state.chat_history:
        st.markdown('<div class="empty-chat">Ask a question</div>', unsafe_allow_html=True)
    else:
        for i, (sender, message) in enumerate(st.session_state.chat_history):
            if sender == 'You':
                st.markdown(f'<div class="user-message">{message}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-message">{message}</div>', unsafe_allow_html=True)
                tts_key = f"tts_btn_{i}"
                if st.button("🔊 Listen", key=tts_key):
                    if hasattr(st.session_state,'best_response_for_audio') and st.session_state.best_response_for_audio:
                        clean_text = clean_text_for_audio(st.session_state.best_response_for_audio)
                    else:
                        clean_text = clean_text_for_audio(message)
                    with st.spinner("Generating audio..."):
                        audio_buffer = generate_audio_from_text(clean_text)
                        if audio_buffer:
                            st.session_state.tts_audio[i] = audio_buffer
                            st.rerun()
                if i in st.session_state.tts_audio:
                    st.audio(st.session_state.tts_audio[i], format='audio/mp3')
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="chat-input-container" style="margin-top: 10px;">', unsafe_allow_html=True)
    input_col, button_col = st.columns([6, 1])
    with input_col:
        user_input = st.text_input(
            "User Question",
            key="user_question_input_chat",
            placeholder="Ask about any children's story...",
            label_visibility="hidden"
        )
    with button_col:
        st.markdown('<div class="send-button">', unsafe_allow_html=True)
        send_button = st.button("Send", key="send_button_chat")
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    if send_button and user_input.strip():
        if not is_valid_input(user_input):
            st.warning("Please enter a valid question (at least 3 characters with alphanumeric content).")
        else:
            original_user_input = user_input
            if st.session_state.get('use_query_optimization', False):
                optimized_queries, _ = optimize_query(user_input)
                if optimized_queries and len(optimized_queries) > 1:
                    search_input = optimized_queries[1]
                else:
                    search_input = user_input
            else:
                search_input = user_input
            selected_pdf = st.session_state.selected_pdf if 'selected_pdf' in st.session_state else "All PDFs"
            selected_search_method = st.session_state.search_method if 'search_method' in st.session_state else "All Methods"
            csv_answer, feedback_msg = findanswer(original_user_input)
            actual_answer = csv_answer if csv_answer is not None else None
            search_data = st.session_state.search_data
            if selected_search_method == "All Methods":
                with st.spinner("Running all search methods..."):
                    multi_results = perform_multi_method_search(
                        search_input,
                        search_data,
                        num_results=num_results,
                        selected_pdf=selected_pdf,
                        use_reranker=st.session_state.get('use_reranker', True)
                    )
                    if not any(results for method, results in multi_results.items() if results):
                        st.warning("No relevant documents found for your query. Please try rephrasing.")
                    else:
                        llm_model = st.session_state.llm_model
                        multi_responses = generate_multi_method_response(
                            search_input,
                            multi_results,
                            llm_model
                        )
                        log_multi_method_results(
                            original_user_input,
                            multi_responses,
                            actual_answer,
                            selected_pdf,
                            llm_model
                        )
                        formatted_response = format_multi_method_response(
                            original_user_input,
                            multi_responses,
                            csv_answer,
                            feedback_msg
                        )
                        st.session_state.chat_history.append(('You', original_user_input))
                        st.session_state.chat_history.append(('Bot', formatted_response))
                        if "hybrid" in multi_results and multi_results["hybrid"]:
                            st.session_state.last_used_chunks = multi_results["hybrid"]
                            st.session_state.top_chunks.append(multi_results["hybrid"][0])
                            st.session_state.selected_chunk_index = multi_results["hybrid"][0].get("index", -1)
                        hybrid_score = multi_responses.get("hybrid", {}).get("bert_score", 0.0)
                        st.session_state.bert_scores.append(hybrid_score)
                        hybrid_rouge_score = multi_responses.get("hybrid", {}).get("rouge_l_score", 0.0)
                        st.session_state.rouge_l_scores.append(hybrid_rouge_score)
                        if "hybrid" in multi_responses:
                            chunk_scores = []
                            rouge_chunk_scores = []
                            for chunk in multi_responses["hybrid"].get("chunks", []):
                                score = compute_bert_score([chunk], multi_responses["hybrid"]["response"])
                                chunk_scores.append(score)
                                rouge_score = compute_rouge_l_score([chunk], multi_responses["hybrid"]["response"])
                                rouge_chunk_scores.append(rouge_score)
                            st.session_state.all_chunk_scores.append(chunk_scores)
                            st.session_state.all_rouge_scores.append(rouge_chunk_scores)
            else:
                with st.spinner(f"Searching using {selected_search_method}..."):
                    if selected_search_method == "Semantic (FAISS)":
                        chunks = semantic_search(
                            search_data['faiss_index'],
                            search_data['texts'],
                            search_data['metadata'],
                            search_input,
                            n_results=num_results,
                            model_name=search_data.get('embedding_model_name')
                        )
                        search_type = "semantic"
                    elif selected_search_method == "Lexical (BM25)":
                        chunks = bm25_search(
                            search_data['bm25_model'],
                            search_data['tokenized_corpus'],
                            search_data['texts'],
                            search_data['metadata'],
                            search_input,
                            n_results=num_results
                        )
                        search_type = "lexical"
                    else:
                        alpha = st.session_state.get('hybrid_alpha', 0.7)
                        n_semantic = st.session_state.get('n_semantic', 7)
                        n_lexical = st.session_state.get('n_lexical', 5)
                        chunking_method = search_data.get('current_chunking_method', 'standard')
                        if chunking_method == "semantic":
                            adjusted_alpha = min(alpha + 0.1, 0.9)
                        else:
                            adjusted_alpha = alpha
                        chunks = hybrid_search(
                            search_data['faiss_index'],
                            search_data['bm25_model'],
                            search_data['tokenized_corpus'],
                            search_data['texts'],
                            search_data['metadata'],
                            search_input,
                            n_semantic=n_semantic,
                            n_lexical=n_lexical,
                            alpha=adjusted_alpha,
                            n_results=num_results,
                            model_name=search_data.get('embedding_model_name')
                        )
                        for chunk in chunks:
                            chunk['chunking_method'] = chunking_method
                            chunk['adjusted_alpha'] = adjusted_alpha
                        search_type = "hybrid"
                    if st.session_state.get('use_reranker', True) and chunks:
                        reranker_model_name = st.session_state.get('reranker_model', get_default_reranker_model())
                        chunks = reranker(search_input, chunks, reranker_model_name)
                    if not chunks:
                        st.warning("No relevant documents found. Try rephrasing.")
                    else:
                        if selected_pdf != "All PDFs":
                            filtered_chunks = filter_chunks_by_pdf(chunks, selected_pdf)
                            if filtered_chunks:
                                chunks_to_use = filtered_chunks
                            else:
                                st.warning(f"No chunks found from {selected_pdf}. Using all retrieved chunks.")
                                chunks_to_use = chunks
                        else:
                            chunks_to_use = chunks
                        if chunks_to_use:
                            st.session_state.last_used_chunks = chunks_to_use
                            st.session_state.top_chunks.append(chunks_to_use[0])
                            st.session_state.selected_chunk_index = chunks_to_use[0].get("index", -1)
                        with st.spinner("Generating response..."):
                            llm_model = st.session_state.llm_model
                            if llm_model == "mistral":
                                query_function = query_mistral
                            else:
                                query_function = query_llama3
                            response = query_function(
                                "Hey there! I'll help you find the answer to your question based on these stories:\n{relevant_documents}\n\nHere's your question: {user_input}\n"
                                "If the answer isn't in the stories, I'll just say 'I don't know'.",search_input,chunks_to_use)
                            chunk_scores, chunks_used = compute_bertscore_with_filter(chunks_to_use, response,selected_pdf)
                            rouge_scores, _ = compute_rougel_with_filter(chunks_to_use, response, selected_pdf)
                            rouge_l_val = max(rouge_scores) if rouge_scores else 0.0
                            if chunk_scores:
                                bert_score_val = max(chunk_scores)
                                most_relevant_chunk_index = chunk_scores.index(bert_score_val)
                                most_relevant_chunk = chunks_used[most_relevant_chunk_index]
                            else:
                                bert_score_val = 0.0
                                most_relevant_chunk = {"text": "No chunk found"}
                            st.session_state.bert_scores.append(bert_score_val)
                            st.session_state.all_chunk_scores.append(chunk_scores)
                            st.session_state.rouge_l_scores.append(rouge_l_val)
                            st.session_state.all_rouge_scores.append(rouge_scores)
                            source_info = format_source_info(most_relevant_chunk)
                            chunking_method = search_data.get('current_chunking_method', 'standard')
                            chunking_display = "Standard" if chunking_method == "standard" else "Semantic"
                            embedding_model = search_data.get('embedding_model_name', 'all-MiniLM-L6-v2')
                            alpha_info = ""
                            reranker_info = ""
                            if search_type == "hybrid" and chunks_to_use and 'adjusted_alpha' in chunks_to_use[0]:
                                alpha_info = f"\n\n<span style='font-weight:bold;'>Alpha:</span> {chunks_to_use[0]['adjusted_alpha']:.2f}"
                            if st.session_state.get('use_reranker', True) and chunks_to_use and 'cross-encoder_score' in \
                                    chunks_to_use[0]:
                                reranker_score = chunks_to_use[0]['cross-encoder_score']
                            search_info = f"\n\n<span style='font-weight:bold;'>Search Method:</span> {selected_search_method} ({chunking_display} chunking, {embedding_model} embeddings){alpha_info}"
                            metrics_info = f"\n\n<span style='font-weight:bold;'>Metrics:</span> BERTScore: {bert_score_val:.4f} | Rouge-L: {rouge_l_val:.4f}"
                            source_display = f"\n\n<span style='font-weight:bold;'>Source:</span> {source_info}"
                            response_answer_rouge_score = None
                            max_chunk_answer_rouge_score = None
                            response_answer_score = None
                            max_chunk_answer_score = None
                            if actual_answer:
                                answer_chunk = [{"text": actual_answer}]
                                response_answer_score = compute_bert_score(answer_chunk, response)
                                response_answer_rouge_score = compute_rouge_l_score(answer_chunk, response)
                                bert_chunk_answer_scores = []
                                rouge_chunk_answer_scores = []
                                for chunk in chunks_used:
                                    chunk_answer_bert_score = compute_bert_score(answer_chunk, chunk["text"])
                                    chunk_answer_rouge_score = compute_rouge_l_score(answer_chunk, chunk["text"])
                                    bert_chunk_answer_scores.append(chunk_answer_bert_score)
                                    rouge_chunk_answer_scores.append(chunk_answer_rouge_score)
                                if bert_chunk_answer_scores:
                                    max_chunk_answer_score = max(bert_chunk_answer_scores)
                                if rouge_chunk_answer_scores:
                                    max_chunk_answer_rouge_score = max(rouge_chunk_answer_scores)
                            csv_notice = ""
                            if csv_answer:
                                csv_notice = "\n\n<span style='color:pink; font-weight:bold;'>Answer from knowledge base: " + csv_answer + "</span>"
                                log_max_bertscore_to_csv(
                                    original_user_input,
                                    response,
                                    actual_answer,
                                    bert_score_val,
                                    response_answer_score,
                                    max_chunk_answer_score,
                                    selected_pdf,
                                    llm_model,
                                    search_type,
                                    rouge_l_val,
                                    response_answer_rouge_score,
                                    max_chunk_answer_rouge_score,
                                    st.session_state.get('use_reranker', True),
                                    st.session_state.get('reranker_model', get_default_reranker_model()),
                                    chunking_method,
                                    st.session_state.get('use_query_optimization'),
                                    embedding_model
                                )
                            st.session_state.chat_history.append(('You', original_user_input))
                            st.session_state.chat_history.append(
                                ('Bot', f"{response}{source_display}{search_info}\n\n{feedback_msg}{csv_notice}")
                            )
            st.rerun()


if __name__ == "__main__":
    main()