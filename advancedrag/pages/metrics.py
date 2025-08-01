import streamlit as st
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from PIL import Image

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from styling.styles import get_css

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
                match = re.search(r'page_(\d+)_img_', img_filename)
                if match:
                    page_num = int(match.group(1))
                    if page_num not in page_images:
                        page_images[page_num] = []
                    page_images[page_num].append(img_path)
            manual_folders[manual_name] = page_images
        return manual_folders
    except Exception as e:
        st.sidebar.error(f"Error scanning  folders: {str(e)}")
        return {}


def filter_chunks_by_pdf(chunks, selected_pdf):
    if not selected_pdf or selected_pdf == "All PDFs":
        return chunks
    filtered_chunks = []
    selected_pdf_base = os.path.splitext(os.path.basename(selected_pdf))[0]
    for chunk in chunks:
        if "metadata" in chunk and "source" in chunk["metadata"]:
            chunk_source = os.path.splitext(os.path.basename(chunk["metadata"]["source"]))[0]
            if chunk_source == selected_pdf_base:
                filtered_chunks.append(chunk)
    return filtered_chunks if filtered_chunks else chunks


def ensure_5_chunks(chunks_to_display, target_count=5):
    if len(chunks_to_display) >= target_count:
        return chunks_to_display[:target_count]
    padded_chunks = chunks_to_display.copy()
    for i in range(len(chunks_to_display), target_count):
        placeholder_chunk = {
            "text": f"[No additional relevant content found - Placeholder {i + 1 - len(chunks_to_display)}]",
            "metadata": {
                "source": "placeholder",
                "page": 0,
                "chunking_method": "placeholder"
            },
            "score": 0.0,
            "index": f"placeholder_{i}",
            "is_placeholder": True
        }
        padded_chunks.append(placeholder_chunk)
    return padded_chunks


def plot_bert_scores(scores):
    fig, ax = plt.subplots(figsize=(10, 5))
    chunks = list(range(1, len(scores) + 1))
    ax.plot(chunks, scores, marker='o', linestyle='-', color='#1f77b4', linewidth=2, markersize=8)
    ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Good (≥0.7)')
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium (≥0.5)')
    ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='Poor (<0.5)')
    ax.set_title('BERTScore Scores by Chunk Number', fontsize=14, pad=20)
    ax.set_xlabel('Chunk Number', fontsize=12)
    ax.set_ylabel('BERTScore Score', fontsize=12)
    ax.set_xticks(chunks)
    ax.set_ylim(0, 1.05)
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.legend(loc='upper right')
    for i, score in enumerate(scores):
        ax.text(chunks[i], score + 0.02, f"{score:.2f}", ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    return fig


def plot_rouge_scores(scores):
    fig, ax = plt.subplots(figsize=(10, 5))
    chunks = list(range(1, len(scores) + 1))
    ax.plot(chunks, scores, marker='s', linestyle='-', color='#2ca02c', linewidth=2, markersize=8)
    ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Good (≥0.5)')
    ax.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='Medium (≥0.3)')
    ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='Poor (<0.1)')
    ax.set_title('Rouge-L Scores by Chunk Number', fontsize=14, pad=20)
    ax.set_xlabel('Chunk Number', fontsize=12)
    ax.set_ylabel('Rouge-L Score', fontsize=12)
    ax.set_xticks(chunks)
    ax.set_ylim(0, 1.05)
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.legend(loc='upper right')
    for i, score in enumerate(scores):
        ax.text(chunks[i], score + 0.02, f"{score:.2f}", ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    return fig


def plot_combined_scores(bert_scores, rouge_scores):
    fig, ax = plt.subplots(figsize=(12, 6))
    chunks = list(range(1, len(bert_scores) + 1))
    ax.plot(chunks, bert_scores, marker='o', linestyle='-', color='#1f77b4', linewidth=2, markersize=8,
            label='BERTScore')
    ax.plot(chunks, rouge_scores, marker='s', linestyle='-', color='#2ca02c', linewidth=2, markersize=8,
            label='Rouge-L')
    ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.2)
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.2)
    ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.2)
    ax.set_title('Comparison of BERTScore and Rouge-L Scores by Chunk', fontsize=14, pad=20)
    ax.set_xlabel('Chunk Number', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xticks(chunks)
    ax.set_ylim(0, 1.05)
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.legend(loc='upper right')
    for i, (bert_score, rouge_score) in enumerate(zip(bert_scores, rouge_scores)):
        ax.text(chunks[i], bert_score + 0.02, f"{bert_score:.2f}", ha='center', va='bottom', fontsize=9,
                color='#1f77b4')
        ax.text(chunks[i], rouge_score - 0.05, f"{rouge_score:.2f}", ha='center', va='top', fontsize=9, color='#2ca02c')
    plt.tight_layout()
    return fig


def find_images_clean(chunk, manual_folders):
    # check empty
    if not manual_folders or not chunk:
        return []
    images = []
    # check if 'metadata' and 'source' exist in the chunk
    if "metadata" in chunk and "source" in chunk["metadata"]:
        source_path = chunk["metadata"]["source"]  # extract the source path
        source_name = os.path.splitext(os.path.basename(source_path))[0]  # get the file name without extension
        if source_name in manual_folders:
            pages = manual_folders[source_name]  # get the pages for this folder
            # check if there is a specific page number in the chunk's metadata
            if "page" in chunk["metadata"]:
                page_num = chunk["metadata"]["page"]  # extract the page number from metadata
                # if the page number exists in the folder, get its images
                if page_num in pages:
                    images = pages[page_num]
                else:
                    # try to get images from adjacent pages (previous or next)
                    for adjacent_page in [page_num - 1, page_num + 1]:
                        if adjacent_page in pages and adjacent_page > 0:  # check if adjacent page exists and is valid
                            images.extend(pages[adjacent_page])  # add images from the adjacent page
                            break
            else:
                # if no specific page, get the first 3 available pages
                available_pages = sorted(pages.keys())[:3]
                for page in available_pages:
                    images.extend(pages[page])  # add images from the first 3 pages
        else:
            # if source_name doesn't match exactly, try matching part of the name
            for folder_name in manual_folders.keys():
                # check if source_name is a substring of folder_name or vice versa
                if source_name in folder_name or folder_name in source_name:
                    pages = manual_folders[folder_name]  # get the pages for the matching folder
                    # if there is a specific page, get the images from that page
                    if "page" in chunk["metadata"]:
                        page_num = chunk["metadata"]["page"]
                        if page_num in pages:
                            images = pages[page_num]
                            break
    return images


def display_chunk_with_images_clean(chunk, chunk_index, rank, bert_score=None, rouge_score=None,
                                    manual_folders=None, metadata=None, show_debug=False):
    is_placeholder = chunk.get("is_placeholder", False)
    with st.container():
        if is_placeholder:
            st.markdown(f"###  Rank #{rank} - No Additional Content Available")
            st.info("This slot is empty because fewer than 5 relevant chunks were found for your query.")
            if show_debug:
                with st.expander(f" Debug Info for Placeholder {rank}", expanded=False):
                    st.write("**Placeholder Chunk Info:**")
                    st.write(f"- Rank: {rank}")
                    st.write(f"- Type: Placeholder")
                    st.write(f"- Reason: Insufficient relevant content")
            st.markdown("---")
            return
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            source_info = format_source_info(chunk)
            chunk_method = "unknown"
            if "metadata" in chunk and chunk["metadata"]:
                chunk_method = chunk["metadata"].get("chunking_method", "unknown")
            score_info = ""
            if 'cross-encoder_score' in chunk:
                score_info = f" (Reranker: {chunk['cross-encoder_score']:.4f})"
            elif 'score' in chunk:
                score_info = f" (Score: {chunk['score']:.4f})"
            st.markdown(f"**Rank #{rank}** - {source_info} [{chunk_method}]{score_info}")
        with col2:
            if bert_score is not None:
                bert_color = 'green' if bert_score > 0.7 else 'orange' if bert_score > 0.5 else 'red'
                st.markdown(f"<span style='color:{bert_color}; font-weight:bold;'>BERTScore: {bert_score:.4f}</span>",unsafe_allow_html=True)
            else:
                st.markdown("BERTScore: N/A")
        with col3:
            if rouge_score is not None:
                rouge_color = 'green' if rouge_score > 0.5 else 'orange' if rouge_score > 0.3 else 'red'
                st.markdown(f"<span style='color:{rouge_color}; font-weight:bold;'>Rouge-L: {rouge_score:.4f}</span>",unsafe_allow_html=True)
            else:
                st.markdown("Rouge-L: N/A")
        chunk_text = chunk.get("text", "No text available")
        text_preview = chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
        with st.expander(f" Chunk {rank} Text ({len(chunk_text)} chars)", expanded=False):
            st.markdown(f'<div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px;">{chunk_text}</div>',unsafe_allow_html=True)
        st.markdown(f'<div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; font-size: 0.9em; margin: 5px 0;">{text_preview}</div>',unsafe_allow_html=True)
        if manual_folders:
            images = find_images_clean(chunk, manual_folders)
            if images:
                st.markdown(f"**{len(images)} image(s) found for Chunk {rank}:**")
                if len(images) == 1:
                    try:
                        img = Image.open(images[0])
                        img_filename = os.path.basename(images[0])
                        page_match = re.search(r'page_(\d+)_img_', img_filename)
                        page_num = page_match.group(1) if page_match else "unknown"
                        pdf_name = os.path.basename(os.path.dirname(images[0]))
                        st.image(img,caption=f"{pdf_name}, Page {page_num}",use_container_width=True)
                    except Exception as e:
                        st.error(f"Error loading image: {str(e)}")
                else:
                    image_cols = st.columns(min(len(images), 3))
                    for i, img_path in enumerate(images):
                        try:
                            img = Image.open(img_path)
                            with image_cols[i % len(image_cols)]:
                                img_filename = os.path.basename(img_path)
                                page_match = re.search(r'page_(\d+)_img_', img_filename)
                                page_num = page_match.group(1) if page_match else "unknown"
                                pdf_name = os.path.basename(os.path.dirname(img_path))
                                st.image(img,caption=f"{pdf_name}, Page {page_num}",use_container_width=True)
                        except Exception as e:
                            with image_cols[i % len(image_cols)]:
                                st.error(f"Error loading image: {str(e)}")
            else:
                st.info(f" No images found for Chunk {rank}")
                if show_debug and "metadata" in chunk and "source" in chunk["metadata"]:
                    source_name = os.path.splitext(os.path.basename(chunk["metadata"]["source"]))[0]
                    st.caption(f"Searched for: {source_name}")
        st.markdown("---")


def display_metrics_dashboard():
    st.set_page_config(page_title=" Analytics", layout="wide")
    st.markdown(get_css(), unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">Metrics Dashboard</h1>', unsafe_allow_html=True)
    nav_cols = st.columns([1, 1, 1, 1])
    with nav_cols[0]:
        if st.button("Back to Chatbot", key="back_btn"):
            st.switch_page("main_menu.py")
    with nav_cols[1]:
        if st.button("Go to Score Logs", key="back_btn2"):
            st.switch_page("pages/knowledge_and_scores_viewer.py")
    with nav_cols[2]:
        if st.button("Metrics for all", key="back_btn3"):
            st.switch_page("pages/metrics_for_all.py")
    if 'chat_history' not in st.session_state:
        st.warning("No chat history found. Please use the chatbot first.")
        return
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('<div class="chunk-header">Analytics for Latest Response</div>', unsafe_allow_html=True)
        if len(st.session_state.chat_history) < 2:
            st.markdown(f'<p style="color:#961233; text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.3);">No bot responses found. Ask a question first.</p>',unsafe_allow_html=True)
            return
        latest_question = None
        latest_response = None
        for i in range(len(st.session_state.chat_history) - 1, -1, -1):
            sender, message = st.session_state.chat_history[i]
            if sender == 'Bot':
                latest_response = message
                if i > 0 and st.session_state.chat_history[i - 1][0] == 'You':
                    latest_question = st.session_state.chat_history[i - 1][1]
                    break
        if not latest_question or not latest_response:
            st.warning("Couldn't find a complete question-answer pair.")
            return
        st.subheader("Latest Conversation")
        st.markdown(f'<div class="user-message">{latest_question}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="bot-message">{latest_response}</div>', unsafe_allow_html=True)
        st.subheader("Response Quality")
        has_bert_scores = 'bert_scores' in st.session_state and st.session_state.bert_scores
        has_rouge_scores = 'rouge_l_scores' in st.session_state and st.session_state.rouge_l_scores
        quality_cols = st.columns([1, 1])
        with quality_cols[0]:
            if has_bert_scores:
                latest_bert_score = st.session_state.bert_scores[-1]
                bert_score_color = 'lightgreen' if latest_bert_score > 0.7 else 'orange' if latest_bert_score > 0.5 else 'pink'
                st.markdown(f'<div style="background-color:{bert_score_color}; color:black; padding: 10px; border-radius: 5px;"><strong>BERTScore:</strong> {latest_bert_score:.4f}</div>',unsafe_allow_html=True )
            else:
                st.info("No BERTScore data available.")
        with quality_cols[1]:
            if has_rouge_scores:
                latest_rouge_score = st.session_state.rouge_l_scores[-1]
                rouge_score_color = 'lightgreen' if latest_rouge_score > 0.5 else 'orange' if latest_rouge_score > 0.3 else 'pink'
                st.markdown(f'<div style="background-color:{rouge_score_color}; color:black; padding: 10px; border-radius: 5px;"><strong>Rouge-L:</strong> {latest_rouge_score:.4f}</div>',unsafe_allow_html=True)
            else:
                st.info("No Rouge-L data available.")
        has_bert_chunk_scores = 'all_chunk_scores' in st.session_state and st.session_state.all_chunk_scores
        has_rouge_chunk_scores = 'all_rouge_scores' in st.session_state and st.session_state.all_rouge_scores
        if has_bert_chunk_scores or has_rouge_chunk_scores:
            st.subheader("Chunk Scores Analysis")
            latest_bert_chunk_scores = st.session_state.all_chunk_scores[-1] if has_bert_chunk_scores else []
            latest_rouge_chunk_scores = st.session_state.all_rouge_scores[-1] if has_rouge_chunk_scores else []
            if has_bert_chunk_scores and has_rouge_chunk_scores:
                min_length = min(len(latest_bert_chunk_scores), len(latest_rouge_chunk_scores))
                if min_length > 0:
                    score_data = {
                        "Chunk": [f"Chunk {i + 1}" for i in range(min_length)],
                        "BERTScore": [f"{score:.4f}" for score in latest_bert_chunk_scores[:min_length]],
                        "BERTScore Quality": ["Good" if score > 0.7 else "Medium" if score > 0.5 else "Poor"
                                         for score in latest_bert_chunk_scores[:min_length]],
                        "Rouge-L": [f"{score:.4f}" for score in latest_rouge_chunk_scores[:min_length]],
                        "Rouge-L Quality": ["Good" if score > 0.5 else "Medium" if score > 0.3 else "Poor"
                                            for score in latest_rouge_chunk_scores[:min_length]]
                    }
                    scores_df = pd.DataFrame(score_data)
                    st.dataframe(scores_df, hide_index=True)
                    viz_tabs = st.tabs(["BERTScore", "Rouge-L", "Combined"])
                    with viz_tabs[0]:
                        bert_fig = plot_bert_scores(latest_bert_chunk_scores[:min_length])
                        st.pyplot(bert_fig)
                    with viz_tabs[1]:
                        rouge_fig = plot_rouge_scores(latest_rouge_chunk_scores[:min_length])
                        st.pyplot(rouge_fig)
                    with viz_tabs[2]:
                        combined_fig = plot_combined_scores(
                            latest_bert_chunk_scores[:min_length],
                            latest_rouge_chunk_scores[:min_length]
                        )
                        st.pyplot(combined_fig)
            elif has_bert_chunk_scores:
                score_data = {"Chunk": [], "BERTScore": [], "Quality": []}
                for i, score in enumerate(latest_bert_chunk_scores):
                    score_data["Chunk"].append(f"Chunk {i + 1}")
                    score_data["BERTScore"].append(f"{score:.4f}")
                    quality = "Good" if score > 0.7 else "Medium" if score > 0.5 else "Poor"
                    score_data["Quality"].append(quality)
                scores_df = pd.DataFrame(score_data)
                st.dataframe(scores_df, hide_index=True)
                fig = plot_bert_scores(latest_bert_chunk_scores)
                st.pyplot(fig)
            elif has_rouge_chunk_scores:
                score_data = {"Chunk": [], "Rouge-L": [], "Quality": []}
                for i, score in enumerate(latest_rouge_chunk_scores):
                    score_data["Chunk"].append(f"Chunk {i + 1}")
                    score_data["Rouge-L"].append(f"{score:.4f}")
                    quality = "Good" if score > 0.5 else "Medium" if score > 0.3 else "Poor"
                    score_data["Quality"].append(quality)
                scores_df = pd.DataFrame(score_data)
                st.dataframe(scores_df, hide_index=True)
                fig = plot_rouge_scores(latest_rouge_chunk_scores)
                st.pyplot(fig)
    with col2:
        chunks_available = False
        chunks_to_display = []
        if 'last_used_chunks' in st.session_state and st.session_state.last_used_chunks:
            chunks_to_display = st.session_state.last_used_chunks
            chunks_available = True
        elif 'retrieved_chunks' in st.session_state and st.session_state.retrieved_chunks:
            chunks_to_display = st.session_state.retrieved_chunks
            chunks_available = True
        elif 'top_chunks' in st.session_state and st.session_state.top_chunks:
            if isinstance(st.session_state.top_chunks, list) and len(st.session_state.top_chunks) > 0:
                chunks_to_display = [st.session_state.top_chunks[-1]]
                chunks_available = True
        if not chunks_available:
            st.warning(" No chunks available. Please ask a question in the chatbot first.")
            return
        selected_pdf = st.session_state.get('selected_pdf', "All PDFs")
        if selected_pdf != "All PDFs":
            filtered_chunks = filter_chunks_by_pdf(chunks_to_display, selected_pdf)
            if filtered_chunks:
                chunks_to_display = filtered_chunks
            else:
                st.warning(f"️ No chunks found from {selected_pdf}. Using all chunks.")
        original_count = len(chunks_to_display)
        exactly_5_chunks = ensure_5_chunks(chunks_to_display, 5)
        latest_bert_chunk_scores = []
        latest_rouge_chunk_scores = []
        if 'all_chunk_scores' in st.session_state and st.session_state.all_chunk_scores:
            scores = st.session_state.all_chunk_scores[-1]
            latest_bert_chunk_scores = scores + [0.0] * (5 - len(scores))
        if 'all_rouge_scores' in st.session_state and st.session_state.all_rouge_scores:
            scores = st.session_state.all_rouge_scores[-1]
            latest_rouge_chunk_scores = scores + [0.0] * (5 - len(scores))
        manual_folders = scan_manual_folders("extracted_images")
        st.session_state.manual_folders = manual_folders
        metadata = None
        if 'search_data' in st.session_state and st.session_state.search_data:
            metadata = st.session_state.search_data.get('metadata')
        for i, chunk in enumerate(exactly_5_chunks):
            chunk_index = chunk.get("index", i)
            bert_score = latest_bert_chunk_scores[i] if i < len(latest_bert_chunk_scores) else 0.0
            rouge_score = latest_rouge_chunk_scores[i] if i < len(latest_rouge_chunk_scores) else 0.0
            display_chunk_with_images_clean(
                chunk=chunk,
                chunk_index=chunk_index,
                rank=i + 1,
                bert_score=bert_score,
                rouge_score=rouge_score,
                manual_folders=manual_folders,
                metadata=metadata
            )
        if chunks_to_display:
            st.markdown("---")
            st.subheader(" Quick Statistics")
            stats_cols = st.columns(4)
            with stats_cols[0]:
                st.metric("Chunks Displayed", "5 ")
            with stats_cols[1]:
                if latest_bert_chunk_scores:
                    real_scores = [s for s in latest_bert_chunk_scores[:original_count] if s > 0]
                    avg_bert = sum(real_scores) / len(real_scores) if real_scores else 0
                    st.metric("Avg BERTScore Score", f"{avg_bert:.3f}")
                else:
                    st.metric("Avg BERTScore Score", "N/A")
            with stats_cols[2]:
                if latest_rouge_chunk_scores:
                    real_scores = [s for s in latest_rouge_chunk_scores[:original_count] if s > 0]
                    avg_rouge = sum(real_scores) / len(real_scores) if real_scores else 0
                    st.metric("Avg Rouge-L Score", f"{avg_rouge:.3f}")
                else:
                    st.metric("Avg Rouge-L Score", "N/A")
            with stats_cols[3]:
                chunks_with_images = 0
                if manual_folders:
                    for chunk in exactly_5_chunks[:original_count]:
                        if not chunk.get("is_placeholder", False):
                            images = find_images_clean(chunk, manual_folders)
                            if images:
                                chunks_with_images += 1

def main():
    display_metrics_dashboard()

if __name__ == "__main__":
    main()