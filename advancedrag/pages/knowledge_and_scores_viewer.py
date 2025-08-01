import streamlit as st
import pandas as pd
import os
import sys


root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from styling.styles import get_css

base_path = "Contents"
csv_file_path = os.path.join(base_path, "file.csv")
scores_log_path = os.path.join(base_path, "scores_log.csv")


def load_csv_data():
    try:
        df = pd.read_csv(csv_file_path, encoding='utf-8-sig',sep=';')
        return df, None
    except FileNotFoundError:
        return None, f"File not found: {csv_file_path}"
    except Exception as e:
        return None, f"Error loading file: {str(e)}"


def load_scores_log():
    try:
        df = pd.read_csv(scores_log_path,encoding='utf-8-sig',sep=';',on_bad_lines='skip',low_memory=False)
        return df, None
    except FileNotFoundError:
        return None, f"Scores log file not found: {scores_log_path}"
    except Exception as e:
        return None, f"Error loading scores log: {str(e)}"


def main():
    st.set_page_config(page_title="Knowledge Base & Scores Viewer", layout="wide")
    st.markdown(get_css(), unsafe_allow_html=True)
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        font-size: 2.2rem;
        margin-bottom: 20px;
        color: #4169E1;
    }
    .edit-form {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .search-box {
        margin-bottom: 20px;
    }
    .button-container {
        display: flex;
        gap: 10px;
    }
    .tab-content {
        padding: 20px 0;
    }
    .score-good {
        color: green;
        font-weight: bold;
    }
    .score-medium {
        color: orange;
        font-weight: bold;
    }
    .score-poor {
        color: red;
        font-weight: bold;
    }
    .metrics-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #666;
        margin-bottom: 2px;
    }
    .metric-score {
        font-size: 1.2rem;
        font-weight: bold;
    }
    .metric-row {
        display: flex;
        justify-content: space-between;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    if 'editing' not in st.session_state:
        st.session_state.editing = False
    if 'edit_index' not in st.session_state:
        st.session_state.edit_index = None
    if 'edit_question' not in st.session_state:
        st.session_state.edit_question = ""
    if 'edit_answer' not in st.session_state:
        st.session_state.edit_answer = ""
    if 'add_mode' not in st.session_state:
        st.session_state.add_mode = False
    if 'filter_text' not in st.session_state:
        st.session_state.filter_text = ""
    if 'scores_filter_text' not in st.session_state:
        st.session_state.scores_filter_text = ""
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "Knowledge Base"
    if 'sort_by' not in st.session_state:
        st.session_state.sort_by = "None"
    if 'sort_order' not in st.session_state:
        st.session_state.sort_order = "Descending"
    if 'score_type_filter' not in st.session_state:
        st.session_state.score_type_filter = "All"
    st.markdown('<h1 class="main-header">Knowledge and Scores Viewer</h1>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Back to Chatbot", key="chatbot_btn", use_container_width=True):
            st.switch_page("main_menu.py")
    with col2:
        if st.button("Metrics", key="metrics_btn", use_container_width=True):
            st.switch_page("pages/metrics.py")
    with col3:
        if st.button("Metrics for all", key="metrics_btn2", use_container_width=True):
            st.switch_page("pages/metrics_for_all.py")
    tab1, tab2 = st.tabs(["Knowledge Base", "Response Scores Log"])
    with tab1:
        st.session_state.active_tab = "Knowledge Base"
        df, error = load_csv_data()
        if error:
            st.error(error)
            st.warning("Please check the file path and try again.")
        else:
            st.subheader("Knowledge Base Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f'<p style="color:red; text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.3);">Total entries: {len(df)}</p>',unsafe_allow_html=True)
            with col2:
                avg_answer_len = df['answer'].str.len().mean()
                st.markdown(f'<p style="color:red; text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.3);">Average answer length: {avg_answer_len:.1f} characters</p>', unsafe_allow_html=True)
            st.subheader("Knowledge Base Entries")
            filter_text = st.text_input("Search questions and answers", value=st.session_state.filter_text,
                                        key="search_box")
            st.session_state.filter_text = filter_text
            if filter_text:
                filtered_df = df[
                    df['question'].str.contains(filter_text, case=False) |
                    df['answer'].str.contains(filter_text, case=False)
                    ]
            else:
                filtered_df = df
            st.dataframe(
                filtered_df,
                column_config={
                    "question": st.column_config.TextColumn("Question"),
                    "answer": st.column_config.TextColumn("Answer"),
                },
                height=400
            )
    with tab2:
        st.session_state.active_tab = "Response Scores Log"
        scores_df, scores_error = load_scores_log()
        if scores_error:
            st.error(scores_error)
            st.warning("Please check the scores log file path and try again.")
        else:
            score_columns = [
                'ResponseChunkBERTScore', 'ResponseAnswerBERTScore', 'ChunkAnswerBERTScore',
                'ResponseChunkRougeL', 'ResponseAnswerRougeL', 'ChunkAnswerRougeL'
            ]
            for col in score_columns:
                if col in scores_df.columns:
                    scores_df[col] = pd.to_numeric(scores_df[col], errors='coerce')
            has_bert = 'ResponseChunkBERTScore' in scores_df.columns
            has_rouge = 'ResponseChunkRougeL' in scores_df.columns
            st.subheader("Scores Log Overview")
            metrics_cols = st.columns(3)
            with metrics_cols[0]:
                st.metric("Total Entries", len(scores_df))
            with metrics_cols[1]:
                if has_bert:
                    chunk_bert_avg = scores_df['ResponseChunkBERTScore'].mean()
                    st.metric("Avg Chunk BERTScore", f"{chunk_bert_avg:.4f}")
                else:
                    st.metric("Avg Chunk BERTScore", "N/A")
            with metrics_cols[2]:
                if has_rouge:
                    chunk_rouge_avg = scores_df['ResponseChunkRougeL'].mean()
                    st.metric("Avg Chunk Rouge-L", f"{chunk_rouge_avg:.4f}")
                else:
                    st.metric("Avg Chunk Rouge-L", "N/A")
            metrics_cols2 = st.columns(3)
            with metrics_cols2[0]:
                if has_bert and 'ResponseAnswerBERTScore' in scores_df.columns:
                    answer_bert_avg = scores_df['ResponseAnswerBERTScore'].dropna().mean()
                    st.metric("Avg Answer BERTScore", f"{answer_bert_avg:.4f}")
                else:
                    st.metric("Avg Answer BERTScore", "N/A")
            with metrics_cols2[1]:
                if has_rouge and 'ResponseAnswerRougeL' in scores_df.columns:
                    answer_rouge_avg = scores_df['ResponseAnswerRougeL'].dropna().mean()
                    st.metric("Avg Answer Rouge-L", f"{answer_rouge_avg:.4f}")
                else:
                    st.metric("Avg Answer Rouge-L", "N/A")
            st.subheader("Search and Filter")
            score_type_options = ["All"]
            if has_bert:
                score_type_options.append("BERTScore")
            if has_rouge:
                score_type_options.append("Rouge-L")
            filter_cols = st.columns([1, 1])
            with filter_cols[0]:
                scores_filter_text = st.text_input("Search in questions or responses",
                                                   value=st.session_state.scores_filter_text,
                                                   key="scores_search_box")
                st.session_state.scores_filter_text = scores_filter_text
            with filter_cols[1]:
                score_type_filter = st.selectbox(
                    "Filter by score type:",
                    options=score_type_options,
                    index=score_type_options.index(st.session_state.score_type_filter)
                )
                st.session_state.score_type_filter = score_type_filter
            filter_cols2 = st.columns([1, 1])
            with filter_cols2[0]:
                if has_bert and (score_type_filter == "All" or score_type_filter == "BERTScore"):
                    bert_score_range = st.select_slider(
                        "Filter by BERTScore range:",
                        options=["All", "Good (≥0.7)", "Medium (≥0.5)", "Poor (<0.5)"],
                        value="All",
                        key="bert_score_range"
                    )
                else:
                    bert_score_range = "All"
            with filter_cols2[1]:
                if has_rouge and (score_type_filter == "All" or score_type_filter == "Rouge-L"):
                    rouge_score_range = st.select_slider(
                        "Filter by Rouge-L range:",
                        options=["All", "Good (≥0.5)", "Medium (≥0.3)", "Poor (<0.3)"],
                        value="All",
                        key="rouge_score_range"
                    )
                else:
                    rouge_score_range = "All"
            if scores_filter_text:
                filtered_scores_df = scores_df[
                    scores_df['question'].str.contains(scores_filter_text, case=False) |
                    scores_df['response'].str.contains(scores_filter_text, case=False)
                    ]
            else:
                filtered_scores_df = scores_df
            if bert_score_range != "All" and 'ResponseChunkBERTScore' in filtered_scores_df.columns:
                if bert_score_range == "Good (≥0.7)":
                    filtered_scores_df = filtered_scores_df[filtered_scores_df['ResponseChunkBERTScore'] >= 0.7]
                elif bert_score_range == "Medium (≥0.5)":
                    filtered_scores_df = filtered_scores_df[(filtered_scores_df['ResponseChunkBERTScore'] >= 0.5) &
                                                            (filtered_scores_df['ResponseChunkBERTScore'] < 0.7)]
                elif bert_score_range == "Poor (<0.5)":
                    filtered_scores_df = filtered_scores_df[filtered_scores_df['ResponseChunkBERTScore'] < 0.5]
            if rouge_score_range != "All" and 'ResponseChunkRougeL' in filtered_scores_df.columns:
                if rouge_score_range == "Good (≥0.5)":
                    filtered_scores_df = filtered_scores_df[filtered_scores_df['ResponseChunkRougeL'] >= 0.5]
                elif rouge_score_range == "Medium (≥0.3)":
                    filtered_scores_df = filtered_scores_df[(filtered_scores_df['ResponseChunkRougeL'] >= 0.3) &
                                                            (filtered_scores_df['ResponseChunkRougeL'] < 0.5)]
                elif rouge_score_range == "Poor (<0.3)":
                    filtered_scores_df = filtered_scores_df[filtered_scores_df['ResponseChunkRougeL'] < 0.3]
            filter_cols3 = st.columns(2)
            with filter_cols3[0]:
                if 'selected_pdf' in scores_df.columns:
                    pdf_options = ["All"] + sorted(scores_df['selected_pdf'].unique().tolist())
                    selected_pdf_filter = st.selectbox("Filter by PDF", options=pdf_options)
                    if selected_pdf_filter != "All":
                        filtered_scores_df = filtered_scores_df[
                            filtered_scores_df['selected_pdf'] == selected_pdf_filter]
            with filter_cols3[1]:
                if 'LLM Model' in scores_df.columns:
                    model_options = ["All"] + sorted(scores_df['LLM Model'].unique().tolist())
                    selected_model_filter = st.selectbox("Filter by LLM Model", options=model_options)
                    if selected_model_filter != "All":
                        filtered_scores_df = filtered_scores_df[
                            filtered_scores_df['LLM Model'] == selected_model_filter]
            sort_cols = st.columns(2)
            with sort_cols[0]:
                sort_options = ["None"]
                for col in score_columns:
                    if col in filtered_scores_df.columns:
                        sort_options.append(col)
                sort_by = st.selectbox("Sort by", options=sort_options,index=min(sort_options.index(st.session_state.sort_by)
                                                 if st.session_state.sort_by in sort_options else 0,len(sort_options) - 1))
                st.session_state.sort_by = sort_by
            with sort_cols[1]:
                sort_order_options = ["Descending", "Ascending"]
                sort_order = st.selectbox("Sort order", options=sort_order_options,index=sort_order_options.index(st.session_state.sort_order))
                st.session_state.sort_order = sort_order
            if sort_by != "None" and sort_by in filtered_scores_df.columns:
                ascending = (sort_order == "Ascending")
                filtered_scores_df = filtered_scores_df.sort_values(by=sort_by, ascending=ascending)
            st.markdown(f''' <div style="
                    background-color: #f8f9fa;
                    border: 1px solid #ced4da;
                    padding: 10px;
                    border-radius: 5px; ">
                    <p style="color:#961233; text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.3);">Showing {len(filtered_scores_df)} out of {len(scores_df)} entries</p>
                </div>''',unsafe_allow_html=True)
            display_cols = ['question', 'response', 'selected_pdf']
            if 'LLM Model' in filtered_scores_df.columns:
                display_cols.append('LLM Model')
            if score_type_filter == "All" or score_type_filter == "BERTScore":
                if 'ResponseChunkBERTScore' in filtered_scores_df.columns:
                    display_cols.append('ResponseChunkBERTScore')
                if 'ResponseAnswerBERTScore' in filtered_scores_df.columns:
                    display_cols.append('ResponseAnswerBERTScore')
                if 'ChunkAnswerBERTScore' in filtered_scores_df.columns:
                    display_cols.append('ChunkAnswerBERTScore')
            if score_type_filter == "All" or score_type_filter == "Rouge-L":
                if 'ResponseChunkRougeL' in filtered_scores_df.columns:
                    display_cols.append('ResponseChunkRougeL')
                if 'ResponseAnswerRougeL' in filtered_scores_df.columns:
                    display_cols.append('ResponseAnswerRougeL')
                if 'ChunkAnswerRougeL' in filtered_scores_df.columns:
                    display_cols.append('ChunkAnswerRougeL')
            display_cols = [col for col in display_cols if col in filtered_scores_df.columns]
            display_df = filtered_scores_df[display_cols].copy()
            if 'response' in display_df.columns:
                display_df['response'] = display_df['response'].apply(lambda x: x[:100] + '...' if len(x) > 100 else x)
            column_config = {
                "question": st.column_config.TextColumn("Question"),
                "response": st.column_config.TextColumn("Response (Truncated)"),
                "selected_pdf": st.column_config.TextColumn("PDF"),
            }
            if 'LLM Model' in display_df.columns:
                column_config["LLM Model"] = st.column_config.TextColumn("LLM Model")
            if 'ResponseChunkBERTScore' in display_df.columns:
                column_config["ResponseChunkBERTScore"] = st.column_config.NumberColumn(
                    "Chunk BERTScore",
                    format="%.4f",
                    help="BERT Score between response and retrieved chunks"
                )
            if 'ResponseAnswerBERTScore' in display_df.columns:
                column_config["ResponseAnswerBERTScore"] = st.column_config.NumberColumn(
                    "Answer BERTScore",
                    format="%.4f",
                    help="BERT Score between response and reference answer"
                )
            if 'ChunkAnswerBERTScore' in display_df.columns:
                column_config["ChunkAnswerBERTScore"] = st.column_config.NumberColumn(
                    "Chunk-Answer BERTScore",
                    format="%.4f",
                    help="BERT Score between retrieved chunks and reference answer"
                )
            if 'ResponseChunkRougeL' in display_df.columns:
                column_config["ResponseChunkRougeL"] = st.column_config.NumberColumn(
                    "Chunk Rouge-L",
                    format="%.4f",
                    help="Rouge-L Score between response and retrieved chunks"
                )
            if 'ResponseAnswerRougeL' in display_df.columns:
                column_config["ResponseAnswerRougeL"] = st.column_config.NumberColumn(
                    "Answer Rouge-L",
                    format="%.4f",
                    help="Rouge-L Score between response and reference answer"
                )
            if 'ChunkAnswerRougeL' in display_df.columns:
                column_config["ChunkAnswerRougeL"] = st.column_config.NumberColumn(
                    "Chunk-Answer Rouge-L",
                    format="%.4f",
                    help="Rouge-L Score between retrieved chunks and reference answer"
                )
            st.dataframe(
                display_df,
                column_config=column_config,
                height=400
            )
            st.subheader("View Full Entry Details")
            if len(filtered_scores_df) > 0:
                entry_indices = list(range(len(filtered_scores_df)))
                selected_entry = st.selectbox(
                    "Select an entry to view full details:",
                    options=entry_indices,
                    format_func=lambda i: f"Entry {i + 1}: {filtered_scores_df.iloc[i]['question'][:50]}..."
                )
                selected_row = filtered_scores_df.iloc[selected_entry]
                with st.expander("Full Entry Details", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### Question")
                        st.markdown(
                            f"<div style='background-color:#f0f2f6; padding:10px; border-radius:5px;'>{selected_row['question']}</div>",unsafe_allow_html=True)
                    with col2:
                        if 'answer' in selected_row and not pd.isna(selected_row['answer']) and selected_row[
                            'answer'] != "Not found":
                            st.markdown("#### Reference Answer")
                            st.markdown(f"<div style='background-color:#f0f2f6; padding:10px; border-radius:5px;'>{selected_row['answer']}</div>",unsafe_allow_html=True)
                    st.markdown("#### Response")
                    st.markdown(f"<div style='background-color:#f0f2f6; padding:10px; border-radius:5px;'>{selected_row['response']}</div>",unsafe_allow_html=True)
                    st.markdown("#### Metadata")
                    meta_cols = st.columns(4)
                    with meta_cols[0]:
                        st.markdown("**PDF:**")
                        st.write(selected_row['selected_pdf'])
                    with meta_cols[1]:
                        if 'LLM Model' in selected_row:
                            st.markdown("**LLM Model:**")
                            st.write(selected_row['LLM Model'])
                    with meta_cols[2]:
                        if 'Search Type' in selected_row:
                            st.markdown("**Search Type:**")
                            st.write(selected_row['Search Type'])
                    st.markdown("#### Scores")
                    score_tabs = st.tabs(["BERTScore Metrics", "Rouge-L Metrics"])
                    with score_tabs[0]:
                        bert_cols = st.columns(3)
                        with bert_cols[0]:
                            if 'ResponseChunkBERTScore' in selected_row:
                                st.markdown("**Response-Chunk Score:**")
                                bert_score_value = float(selected_row['ResponseChunkBERTScore'])
                                bert_score_class = "score-good" if bert_score_value >= 0.7 else "score-medium" if bert_score_value >= 0.5 else "score-poor"
                                st.markdown(f"<span class='{bert_score_class}'>{bert_score_value:.4f}</span>",unsafe_allow_html=True)
                                st.markdown(f'<p class="metric-label">Similarity between the response and retrieved chunks</p>',unsafe_allow_html=True)
                        with bert_cols[1]:
                            if 'ResponseAnswerBERTScore' in selected_row and not pd.isna(
                                    selected_row['ResponseAnswerBERTScore']):
                                st.markdown("**Response-Answer Score:**")
                                resp_ans_bert = float(selected_row['ResponseAnswerBERTScore'])
                                resp_ans_bert_class = "score-good" if resp_ans_bert >= 0.7 else "score-medium" if resp_ans_bert >= 0.5 else "score-poor"
                                st.markdown(f"<span class='{resp_ans_bert_class}'>{resp_ans_bert:.4f}</span>",unsafe_allow_html=True)
                                st.markdown(f'<p class="metric-label">Similarity between the response and reference answer</p>',unsafe_allow_html=True)
                        with bert_cols[2]:
                            if 'ChunkAnswerBERTScore' in selected_row and not pd.isna(
                                    selected_row['ChunkAnswerBERTScore']):
                                st.markdown("**Chunk-Answer Score:**")
                                chunk_ans_bert = float(selected_row['ChunkAnswerBERTScore'])
                                chunk_ans_bert_class = "score-good" if chunk_ans_bert >= 0.7 else "score-medium" if chunk_ans_bert >= 0.5 else "score-poor"
                                st.markdown(f"<span class='{chunk_ans_bert_class}'>{chunk_ans_bert:.4f}</span>",unsafe_allow_html=True)
                                st.markdown(f'<p class="metric-label">Similarity between the retrieved chunks and reference answer</p>', unsafe_allow_html=True)
                    with score_tabs[1]:
                        rouge_cols = st.columns(3)
                        with rouge_cols[0]:
                            if 'ResponseChunkRougeL' in selected_row and not pd.isna(
                                    selected_row['ResponseChunkRougeL']):
                                st.markdown("**Response-Chunk Score:**")
                                rouge_score_value = float(selected_row['ResponseChunkRougeL'])
                                rouge_score_class = "score-good" if rouge_score_value >= 0.5 else "score-medium" if rouge_score_value >= 0.3 else "score-poor"
                                st.markdown(f"<span class='{rouge_score_class}'>{rouge_score_value:.4f}</span>",unsafe_allow_html=True)
                                st.markdown(f'<p class="metric-label">Sequence similarity between the response and retrieved chunks</p>',unsafe_allow_html=True)
                        with rouge_cols[1]:
                            if 'ResponseAnswerRougeL' in selected_row and not pd.isna(
                                    selected_row['ResponseAnswerRougeL']):
                                st.markdown("**Response-Answer Score:**")
                                resp_ans_rouge = float(selected_row['ResponseAnswerRougeL'])
                                resp_ans_rouge_class = "score-good" if resp_ans_rouge >= 0.5 else "score-medium" if resp_ans_rouge >= 0.3 else "score-poor"
                                st.markdown(f"<span class='{resp_ans_rouge_class}'>{resp_ans_rouge:.4f}</span>",unsafe_allow_html=True)
                                st.markdown(f'<p class="metric-label">Sequence similarity between the response and reference answer</p>',unsafe_allow_html=True)
                        with rouge_cols[2]:
                            if 'ChunkAnswerRougeL' in selected_row and not pd.isna(selected_row['ChunkAnswerRougeL']):
                                st.markdown("**Chunk-Answer Score:**")
                                chunk_ans_rouge = float(selected_row['ChunkAnswerRougeL'])
                                chunk_ans_rouge_class = "score-good" if chunk_ans_rouge >= 0.5 else "score-medium" if chunk_ans_rouge >= 0.3 else "score-poor"
                                st.markdown(f"<span class='{chunk_ans_rouge_class}'>{chunk_ans_rouge:.4f}</span>",unsafe_allow_html=True)
                                st.markdown(f'<p class="metric-label">Sequence similarity between the retrieved chunks and reference answer</p>',unsafe_allow_html=True)

if __name__ == "__main__":
    main()