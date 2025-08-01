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


CHUNKING_METHOD = "semantic"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
USE_QUERY_OPTIMIZATION = True
USE_RERANKER = True
HYBRID_ALPHA = 0.7
N_SEMANTIC = 7
N_LEXICAL = 5
LLM_MODEL = "llama3"


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


def findanswer(user_question, user_answer=None):
    csv_file_path = os.path.join(
        'Contents',
        "file.csv")
    try:
        df = pd.read_csv(csv_file_path, encoding='utf-8-sig', sep=';')
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


def scan_manual_folders(images_folder="extracted_images"):
    manual_folders = {}
    if not os.path.exists(images_folder):
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
                match = re.search(r'page_(\d+)_img_', img_filename)
                if match:
                    page_num = int(match.group(1))
                    if page_num not in page_images:
                        page_images[page_num] = []
                    page_images[page_num].append(img_path)
            manual_folders[manual_name] = page_images
        return manual_folders
    except Exception as e:
        st.error(f"Error scanning folders: {str(e)}")
        return {}


def main():
    base_path = "Contents"
    st.set_page_config(page_title="Chatbot", layout="wide", initial_sidebar_state="collapsed")
    st.markdown(get_css(), unsafe_allow_html=True)
    for key, default_value in {
        'chat_history': [], 'bert_scores': [], 'all_chunk_scores': [],
        'top_chunks': [], 'selected_chunk_index': -1, 'displayed_chunk_scores': False,
        'original_documents': [], 'tts_audio': {}, 'llm_model': LLM_MODEL,
        'uploaded_files_info': [], 'chunking_method': CHUNKING_METHOD,
        'embedding_model': EMBEDDING_MODEL, 'search_data': None,
        'last_used_chunks': [],
        'rouge_l_scores': [], 'all_rouge_scores': [],
        'use_query_optimization': USE_QUERY_OPTIMIZATION,
        'use_reranker': USE_RERANKER,
        'hybrid_alpha': HYBRID_ALPHA,
        'n_semantic': N_SEMANTIC,
        'n_lexical': N_LEXICAL,
        'reranker_model': get_default_reranker_model()
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

    def ensure_search_data_loaded():
        try:
            faiss_semantic_path, _, _, _ = get_faiss_file_paths(CHUNKING_METHOD, EMBEDDING_MODEL)
            bm25_semantic_path, _, _, _ = get_bm25_file_paths(CHUNKING_METHOD, EMBEDDING_MODEL)
            semantic_exists = os.path.exists(faiss_semantic_path) and os.path.exists(bm25_semantic_path)
            if not semantic_exists:
                st.info(" First time setup: Creating search indices from your PDF files...")
                set_embedding_model(EMBEDDING_MODEL)
                all_files = get_all_files_in_folder(folder_path)
                if not all_files:
                    st.error("No files found in the folder.")
                    return False
                with st.spinner("Processing PDF documents..."):
                    all_documents_with_pages, file_sources = [], []
                    for file_path in all_files:
                        file_docs = process_files(file_path)
                        if file_docs:
                            all_documents_with_pages.append(file_docs)
                            file_sources.append(file_path)
                if not all_documents_with_pages:
                    st.error("No documents could be processed.")
                    return False
                with st.spinner(" Creating semantic chunks and embeddings..."):
                    try:
                        chunks, metadata = chunk_documents_semantic(all_documents_with_pages, file_sources)
                        embeddings = batch_generate_embeddings(chunks, model_name=EMBEDDING_MODEL)
                        index = create_faiss_index(embeddings, embeddings.shape[1])
                        save_faiss_data(index, embeddings, chunks, metadata, chunking_method=CHUNKING_METHOD,
                                        model_name=EMBEDDING_MODEL)
                        bm25_model, tokenized_corpus = create_bm25_index(chunks)
                        save_data(bm25_model, tokenized_corpus, chunks, metadata, CHUNKING_METHOD, EMBEDDING_MODEL)
                        st.success(" Semantic chunking completed!")
                    except Exception as e:
                        st.warning(f"Semantic chunking failed: {str(e)}")
                        st.info("Using standard chunking as fallback...")
                        chunks, metadata = chunk_documents(all_documents_with_pages, file_sources)
                        embeddings = batch_generate_embeddings(chunks, model_name=EMBEDDING_MODEL)
                        index = create_faiss_index(embeddings, embeddings.shape[1])
                        save_faiss_data(index, embeddings, chunks, metadata, chunking_method=CHUNKING_METHOD,
                                        model_name=EMBEDDING_MODEL)
                        bm25_model, tokenized_corpus = create_bm25_index(chunks)
                        save_data(bm25_model, tokenized_corpus, chunks, metadata, CHUNKING_METHOD, EMBEDDING_MODEL)
                st.success(" Search indices created successfully!")
            with st.spinner(f"Loading {CHUNKING_METHOD} chunking data..."):
                set_embedding_model(EMBEDDING_MODEL)
                index, embeddings, texts, metadata = load_faiss_data(CHUNKING_METHOD, EMBEDDING_MODEL)
                bm25_model, tokenized_corpus, bm25_texts, bm25_metadata = load_data(CHUNKING_METHOD, EMBEDDING_MODEL)
                if index is not None and texts and bm25_model and tokenized_corpus:
                    final_texts = texts if bm25_texts is None else bm25_texts
                    final_metadata = metadata if bm25_metadata is None else bm25_metadata
                    st.session_state.search_data = {
                        'faiss_index': index,
                        'texts': final_texts,
                        'metadata': final_metadata,
                        'bm25_model': bm25_model,
                        'tokenized_corpus': tokenized_corpus,
                        'current_chunking_method': CHUNKING_METHOD,
                        'embeddings': embeddings,
                        'embedding_model_name': EMBEDDING_MODEL
                    }
                    st.session_state.pdf_files = [f for f in get_all_files_in_folder(folder_path) if f.lower().endswith('.pdf')]
                    st.session_state.manual_folders = scan_manual_folders(images_output_folder)
                    images_already_extracted = check_images_extracted(folder_path, images_output_folder)
                    if not images_already_extracted:
                        with st.spinner(" Extracting images from PDFs..."):
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
                            'displayed_chunk_scores', 'tts_audio', 'last_used_chunks', 'rouge_l_scores',
                            'all_rouge_scores']:
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
                if st.button("🔊 Listen to Response", key=tts_key):
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
            if USE_QUERY_OPTIMIZATION:
                optimized_queries, _ = optimize_query(user_input)
                if optimized_queries and len(optimized_queries) > 1:
                    search_input = optimized_queries[1]
                else:
                    search_input = user_input
            else:
                search_input = user_input
            csv_answer, feedback_msg = findanswer(original_user_input)
            actual_answer = csv_answer if csv_answer is not None else None
            search_data = st.session_state.search_data
            with st.spinner(" Running hybrid search..."):
                alpha = HYBRID_ALPHA
                n_semantic = N_SEMANTIC
                n_lexical = N_LEXICAL
                chunking_method = search_data.get('current_chunking_method', CHUNKING_METHOD)
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
                if USE_RERANKER and chunks:
                    reranker_model_name = get_default_reranker_model()
                    chunks = reranker(search_input, chunks, reranker_model_name)
                if not chunks:
                    st.warning("No relevant documents found. Try rephrasing.")
                else:
                    chunks_to_use = chunks
                    if chunks_to_use:
                        st.session_state.last_used_chunks = chunks_to_use
                        st.session_state.top_chunks.append(chunks_to_use[0])
                        st.session_state.selected_chunk_index = chunks_to_use[0].get("index", -1)
                    with st.spinner("Generating response..."):
                        llm_model = LLM_MODEL
                        if llm_model == "mistral":
                            query_function = query_mistral
                        else:
                            query_function = query_llama3
                        response = query_function(
                            "Hey there! I'll help you find the answer to your question based on these stories:\n{relevant_documents}\n\nHere's your question: {user_input}\n"
                            "If the answer isn't in the stories, I'll just say 'I don't know'.",
                            search_input,
                            chunks_to_use
                        )
                        chunk_scores, chunks_used = compute_bertscore_with_filter(chunks_to_use, response, "All PDFs")
                        rouge_scores, _ = compute_rougel_with_filter(chunks_to_use, response, "All PDFs")
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
                        chunking_display = "Semantic"
                        embedding_model = search_data.get('embedding_model_name', EMBEDDING_MODEL)
                        alpha_info = f"\n\n<span style='font-weight:bold;'>Alpha:</span> {adjusted_alpha:.2f}"
                        search_info = f"\n\n<span style='font-weight:bold;'>Search Method:</span> Hybrid ({chunking_display} chunking, {embedding_model} embeddings){alpha_info}"
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
                                "All PDFs",
                                llm_model,
                                f"hybrid",
                                rouge_l_val,
                                response_answer_rouge_score,
                                max_chunk_answer_rouge_score,
                                USE_RERANKER,
                                get_default_reranker_model(),
                                chunking_method,
                                USE_QUERY_OPTIMIZATION,
                                embedding_model
                            )
                        st.session_state.chat_history.append(('You', original_user_input))
                        st.session_state.chat_history.append(
                            ('Bot',
                             f"{response}{source_display}{search_info}\n\n{feedback_msg}{csv_notice}")
                        )
            st.rerun()


if __name__ == "__main__":
    main()