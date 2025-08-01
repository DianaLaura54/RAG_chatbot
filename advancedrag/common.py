#writes everything in the csv file
import csv
import os

import numpy as np
import pandas as pd
import torch
from fastapi import requests
from bert_score import score as bert_score
from rouge_score import rouge_scorer

from embeddings import embedding_model


def log_max_bertscore_to_csv(question, response, actual_answer, max_bert_score,
                             response_answer_bert_score=None, max_chunk_answer_bert_score=None,
                             selected_pdf=None, llm_model=None, search_type=None,
                             max_rouge_score=None, response_answer_rouge_score=None,
                             max_chunk_answer_rouge_score=None,
                             use_reranker=None, reranker_model=None, chunking_method=None,
                             query_optimization=None, embedding_model=None,
                             base_filepath="Contents"):
    filepath = os.path.join(base_filepath, "scores_log.csv")
    file_exists = os.path.exists(filepath)
    try:
        if max_rouge_score is not None and isinstance(max_rouge_score, str):
            max_rouge_score = float(max_rouge_score)
        if response_answer_rouge_score is not None and isinstance(response_answer_rouge_score, str):
            response_answer_rouge_score = float(response_answer_rouge_score)
        if max_chunk_answer_rouge_score is not None and isinstance(max_chunk_answer_rouge_score, str):
            max_chunk_answer_rouge_score = float(max_chunk_answer_rouge_score)
    except ValueError:
        if isinstance(max_rouge_score, str):
            max_rouge_score = None
        if isinstance(response_answer_rouge_score, str):
            response_answer_rouge_score = None
        if isinstance(max_chunk_answer_rouge_score, str):
            max_chunk_answer_rouge_score = None
    with open(filepath, "a", newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f, delimiter=';')
        if not file_exists:
            writer.writerow([
                "question", "response", "answer", "selected_pdf", "LLM Model", "Search Type",
                "ResponseChunkBERTScore", "ResponseChunkRougeL",
                "ResponseAnswerBERTScore", "ResponseAnswerRougeL",
                "ChunkAnswerBERTScore", "ChunkAnswerRougeL",
                "Reranker Used", "Reranker Model", "Chunking Method", "QueryOptimization", "Embedding Model"
            ])
        writer.writerow([
            question,
            response,
            actual_answer if actual_answer else "Not found",
            selected_pdf if selected_pdf else "All PDFs",
            llm_model if llm_model else "llama3",
            search_type,
            f"{max_bert_score:.4f}",
            f"{max_rouge_score:.4f}" if max_rouge_score is not None else "N/A",
            f"{response_answer_bert_score:.4f}" if response_answer_bert_score is not None else "N/A",
            f"{response_answer_rouge_score:.4f}" if response_answer_rouge_score is not None else "N/A",
            f"{max_chunk_answer_bert_score:.4f}" if max_chunk_answer_bert_score is not None else "N/A",
            f"{max_chunk_answer_rouge_score:.4f}" if max_chunk_answer_rouge_score is not None else "N/A",
            "Yes" if use_reranker else "No",
            reranker_model if reranker_model else "N/A",
            chunking_method if chunking_method else "standard",
            "Yes" if query_optimization else "No",
            embedding_model if embedding_model else "all-MiniLM-L6-v2"
        ])

#generate a random question from the csv
def get_random_question():
    csv_file_path = os.path.join("Contents","file.csv")
    try:
        df = pd.read_csv(csv_file_path, encoding='utf-8-sig',sep=';')
        if not df.empty and 'question' in df.columns:
            random_idx = np.random.randint(0, len(df))
            return df.iloc[random_idx]['question']
        else:
            return "No questions found in file.csv"
    except Exception as e:
        return f"Error loading question: {str(e)}"



# extracts the source file name and the page number from metadata
def format_source_info(chunk):
    source_info = "Unknown source"
    if "metadata" in chunk:
        metadata = chunk["metadata"]
        if "source" in metadata:
            source_name = os.path.basename(metadata["source"])
            source_info = f"{source_name}"
        if "page" in metadata:
            page_num = metadata["page"]
            source_info += f", Page {page_num}"
    return source_info

#valid input
def is_valid_input(user_input):
    if not user_input.strip():
        return False
    if len(user_input) < 3:
        return False
    if not any(char.isalnum() for char in user_input):
        return False
    return True


#checks whether images have been extracted for each PDF in a given folder and verifies that those images exist in a specified images folder
#takes each pdf from the manual folder, check if it's a pdf,checks the image folder,then for each pdf verifies if the output folder for the extracted images
#exist
def check_images_extracted(manuals_path, images_folder):
    pdfs = [f for f in os.listdir(manuals_path) if f.lower().endswith('.pdf')]
    if not pdfs:
        return True
    if not os.path.exists(images_folder):
        return False
    for pdf in pdfs:
        pdf_name = os.path.splitext(pdf)[0]
        pdf_images_folder = os.path.join(images_folder, pdf_name)
        if not os.path.exists(pdf_images_folder):
            return False
        #checks if the image folder for that pdf is empty or not
        image_files = [f for f in os.listdir(pdf_images_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not image_files:
            return False
    return True


def optimize_query(query, model_name=None):
    from embeddings import get_embedding_model, set_embedding_model

    basic_variations = [
        query,
        f"information about {query}",
        f"explain {query}",
        f"details regarding {query}"
    ]
    if model_name:
        set_embedding_model(model_name)
    current_model = get_embedding_model()
    if current_model is None:
        set_embedding_model('all-MiniLM-L6-v2')
        current_model = get_embedding_model()
    if current_model is None:
        raise ValueError("No embedding model available. Please ensure embeddings module is properly initialized.")
    all_embeddings = current_model.encode(basic_variations)
    return basic_variations, all_embeddings

