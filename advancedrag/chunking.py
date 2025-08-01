import os
import re
from collections import Counter

import numpy as np
import spacy

from embeddings import DEFAULT_EMBEDDING_MODEL

nlp = spacy.load("en_core_web_sm")

def split_into_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def chunk_documents_semantic(documents_with_pages, file_sources, embedding_model=None):
    # if no embedding model is provided, attempt to load a default one
    if embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            embedding_model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
        except (ImportError, NameError):
            # if loading fails, fall back to standard chunking
            return chunk_documents(documents_with_pages, file_sources)
    # initialize lists to hold text chunks and metadata
    all_chunks, all_metadata = [], []
    doc_index = 0  # index counter for chunks
    # iterate through each document and its corresponding source file
    for src_idx, doc_tuples in enumerate(documents_with_pages):
        source_file = file_sources[src_idx]  # get source file name
        # iterate through content and page number pairs
        for page_content, page_num in doc_tuples:
            # skip empty content
            if not page_content.strip():
                continue
            # normalize whitespace in the content
            content = re.sub(r'\s+', ' ', page_content).strip()
            # split content into sentences using spaCy
            try:
                sentences = split_into_sentences(content)
            except Exception:
                # fallback sentence splitter using punctuation
                sentences = re.split(r'(?<=[.!?])\s+', content)
            # skip if no sentences found
            if not sentences:
                continue
            # attempt to generate sentence embeddings
            try:
                sentence_embeddings = embedding_model.encode(sentences)
            except Exception as e:
                print(f"Error generating embeddings: {str(e)}")
                sentence_embeddings = None
            # define chunk size limits
            min_chunk_size, max_chunk_size = 50, 250
            # define a similarity threshold for semantic cohesion
            similarity_threshold = 0.5
            i = 0  # initialize sentence index
            # iterate through sentences to build chunks
            while i < len(sentences):
                # skip sentences that don't start with an uppercase letter
                while i < len(sentences):
                    if sentences[i] and re.match(r'^[A-Z]', sentences[i]):
                        break
                    i += 1
                if i >= len(sentences):
                    break
                # start a new chunk with the current sentence
                current_chunk_sentences = [sentences[i]]
                current_words = len(sentences[i].split())
                current_idx = i
                i += 1
                # continue adding sentences to the chunk
                while i < len(sentences):
                    sentence = sentences[i]
                    sentence_words = len(sentence.split())
                    # stop if the chunk would exceed the max size
                    if current_words + sentence_words > max_chunk_size:
                        break
                    # if embeddings exist, check semantic similarity
                    if sentence_embeddings is not None:
                        try:
                            prev_embedding = sentence_embeddings[i - 1]
                            curr_embedding = sentence_embeddings[i]
                            # compute cosine similarity
                            similarity = np.dot(prev_embedding, curr_embedding) / (
                                np.linalg.norm(prev_embedding) * np.linalg.norm(curr_embedding) + 1e-8)
                            # if similarity drops below threshold and chunk is large enough, end the chunk
                            if similarity < similarity_threshold and current_words >= min_chunk_size:
                                break
                        except Exception:
                            pass  # ignore similarity errors
                    # add sentence to current chunk
                    current_chunk_sentences.append(sentence)
                    current_words += sentence_words
                    i += 1
                # if chunk meets minimum word count, save it
                if current_chunk_sentences and current_words >= min_chunk_size:
                    chunk_text = " ".join(current_chunk_sentences)
                    all_chunks.append(chunk_text)
                    # save metadata for the chunk
                    all_metadata.append({
                        "source": source_file,
                        "page": page_num,
                        "doc_index": doc_index,
                        "sentences": len(current_chunk_sentences),
                        "words": current_words,
                        "chunking_method": "semantic"
                    })
                    doc_index += 1  # increment chunk index
    # return all generated chunks and their metadata
    return all_chunks, all_metadata


def chunk_documents(documents_with_pages, file_sources):
    # initialize lists to store chunks and metadata
    all_chunks, all_metadata = [], []
    # initialize a counter for chunk indexing
    doc_index = 0
    # loop through each document and its corresponding source file
    for src_idx, doc_tuples in enumerate(documents_with_pages):
        source_file = file_sources[src_idx]  # get the source filename
        # process each content and page_info pair in the document
        for content, page_info in doc_tuples:
            # normalize whitespace in the content
            content = re.sub(r'\s+', ' ', content).strip()
            # try to split the content into sentences using spaCy
            try:
                sentences = split_into_sentences(content)
            except Exception:
                # fallback method for splitting text into sentences
                raw_sentences = re.split(r'\.(?=\s|$)', content)
                sentences = [s.strip() + '.' if not s.strip().endswith('.') else s.strip() for s in raw_sentences if s.strip()]
            # skip if no sentences found
            if not sentences:
                continue
            # initialize list to store page number for each sentence
            sentence_pages = []
            # if page_info is a list of markers indicating position and page number
            if isinstance(page_info, list):
                page_markers = page_info
                content_length = len(content)
                sentence_start_pos = 0
                # assign a page number to each sentence based on its position
                for sentence in sentences:
                    sentence_length = len(sentence)
                    sentence_middle_pos = sentence_start_pos + (sentence_length // 2)
                    page_num = 1  # default page number
                    # find the correct page number based on midpoint position
                    for pos, page in page_markers:
                        if sentence_middle_pos >= pos:
                            page_num = page
                        else:
                            break
                    sentence_pages.append(page_num)
                    sentence_start_pos += sentence_length + 1  # move to next sentence position
            else:
                # if page_info is a single number, use it for all sentences
                page_num = page_info
                sentence_pages = [page_num] * len(sentences)
            # initialize variables to build a chunk
            current_chunk, current_sentences, current_pages = "", [], []
            min_chunk_words, max_chunk_words, ideal_chunk_words = 50, 250, 150
            # iterate over each sentence
            for i, sentence in enumerate(sentences):
                sentence_words = len(sentence.split())  # number of words in sentence
                current_words = len(current_chunk.split()) if current_chunk else 0  # words in current chunk
                # if adding this sentence would exceed max chunk size
                if current_words > 0 and current_words + sentence_words > max_chunk_words:
                    # if current chunk is already big enough, finalize it
                    if current_words >= min_chunk_words:
                        all_chunks.append(current_chunk)
                        # find the most common page number in this chunk
                        if current_pages:
                            page_counter = Counter(current_pages)
                            most_common_page = page_counter.most_common(1)[0][0]
                        else:
                            most_common_page = 1
                        # add metadata for the chunk
                        all_metadata.append({
                            "source": source_file,
                            "page": most_common_page,
                            "doc_index": doc_index,
                            "sentences": len(current_sentences),
                            "words": current_words,
                            "chunking_method": "standard"
                        })
                        doc_index += 1
                        # start a new chunk with the current sentence
                        current_chunk = sentence
                        current_sentences = [sentence]
                        current_pages = [sentence_pages[i]]
                    else:
                        # if chunk is too small, add sentence anyway
                        current_chunk += " " + sentence if current_chunk else sentence
                        current_sentences.append(sentence)
                        current_pages.append(sentence_pages[i])
                else:
                    # add sentence to current chunk
                    current_chunk = current_chunk + " " + sentence if current_chunk else sentence
                    current_sentences.append(sentence)
                    current_pages.append(sentence_pages[i])
                # if chunk size exceeds ideal length and ends with period, finalize it
                if len(current_chunk.split()) > ideal_chunk_words and current_chunk.endswith('.'):
                    all_chunks.append(current_chunk)
                    # determine most common page number in the chunk
                    if current_pages:
                        page_counter = Counter(current_pages)
                        most_common_page = page_counter.most_common(1)[0][0]
                    else:
                        most_common_page = 1
                    # add metadata for the chunk
                    all_metadata.append({
                        "source": source_file,
                        "page": most_common_page,
                        "doc_index": doc_index,
                        "sentences": len(current_sentences),
                        "words": len(current_chunk.split()),
                        "chunking_method": "standard"
                    })
                    doc_index += 1
                    # reset the chunk builder
                    current_chunk, current_sentences, current_pages = "", [], []
            # after finishing all sentences, save any leftover chunk
            if current_chunk:
                all_chunks.append(current_chunk)
                # determine most common page number in the chunk
                if current_pages:
                    page_counter = Counter(current_pages)
                    most_common_page = page_counter.most_common(1)[0][0]
                else:
                    most_common_page = 1
                # add metadata for the final chunk
                all_metadata.append({
                    "source": source_file,
                    "page": most_common_page,
                    "doc_index": doc_index,
                    "sentences": len(current_sentences),
                    "words": len(current_chunk.split()),
                    "chunking_method": "standard"
                })
                doc_index += 1
    # return the list of text chunks and their metadata
    return all_chunks, all_metadata
