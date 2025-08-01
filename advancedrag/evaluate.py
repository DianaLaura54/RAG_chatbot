import os

from bert_score import score as bert_score
from rouge_score import rouge_scorer
import torch

def compute_bert_score(relevant_chunks, response):
    try:
        if not relevant_chunks or not response:
            return 0.0
        references = [chunk["text"].strip() for chunk in relevant_chunks if
                      isinstance(chunk, dict) and "text" in chunk and len(chunk["text"].strip()) >= 10]
        if len(references) == 0 or len(response.strip()) < 10:
            return 0.0
        candidates = [response] * len(references)
        _, _, F1 = bert_score(candidates, references, lang='en', device='cuda' if torch.cuda.is_available() else 'cpu')
        return max(F1).item()
    except Exception as e:
        print(f"Error computing BERTScore: {e}")
        return 0.0



#take the chunks from that pdf and compute the bertscore only for them
def compute_bertscore_with_filter(chunks, response, selected_pdf=None):
    if selected_pdf and selected_pdf != "All PDFs":
        filtered_chunks = filter_chunks_by_pdf(chunks, selected_pdf)
        if filtered_chunks:
            chunks_to_use = filtered_chunks
        else:
            chunks_to_use = chunks
    else:
        chunks_to_use = chunks
    chunk_scores = []
    for chunk in chunks_to_use:
        score = compute_bert_score([chunk], response)
        chunk_scores.append(score)
    return chunk_scores, chunks_to_use


def compute_rouge_l_score(relevant_chunks, response):
    try:
        if not relevant_chunks or not response:
            return 0.0
        references = [chunk["text"].strip() for chunk in relevant_chunks if
                      isinstance(chunk, dict) and "text" in chunk and len(chunk["text"].strip()) >= 10]
        if len(references) == 0 or len(response.strip()) < 10:
            return 0.0
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = []
        for reference in references:
            score = scorer.score(reference, response)
            scores.append(score['rougeL'].fmeasure)  # Use F1 measure
        return max(scores) if scores else 0.0
    except Exception as e:
        print(f"Error computing Rouge-L score: {e}")
        return 0.0


def compute_rougel_with_filter(chunks, response, selected_pdf=None):
    if selected_pdf and selected_pdf != "All PDFs":
        filtered_chunks = filter_chunks_by_pdf(chunks, selected_pdf)
        if filtered_chunks:
            chunks_to_use = filtered_chunks
        else:
            chunks_to_use = chunks
    else:
        chunks_to_use = chunks
    chunk_scores = []
    for chunk in chunks_to_use:
        score = compute_rouge_l_score([chunk], response)
        chunk_scores.append(score)
    return chunk_scores, chunks_to_use

#take the chunks from that pdf,only from that specific pdf
def filter_chunks_by_pdf(chunks, selected_pdf):
    if not selected_pdf or selected_pdf == "All PDFs":
        return chunks
    selected_pdf_base = os.path.splitext(selected_pdf)[0]
    filtered_chunks = []
    for chunk in chunks:
        if "metadata" in chunk and "source" in chunk["metadata"]:
            chunk_source = os.path.basename(chunk["metadata"]["source"])
            if selected_pdf_base in chunk_source:
                filtered_chunks.append(chunk)
    return filtered_chunks