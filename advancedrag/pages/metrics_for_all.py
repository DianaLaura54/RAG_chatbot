import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys


root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from styling.styles import get_css

base_path = "Contents"
scores_log_path = os.path.join(base_path, "scores_log.csv")


def load_scores_log():
    try:
        df = pd.read_csv(scores_log_path,encoding='utf-8-sig',sep=';',on_bad_lines='skip',low_memory=False)
        score_columns = [
            'ResponseChunkBERTScore', 'ResponseAnswerBERTScore', 'ChunkAnswerBERTScore',
            'ResponseChunkRougeL', 'ResponseAnswerRougeL', 'ChunkAnswerRougeL'
        ]
        for col in score_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df, None
    except FileNotFoundError:
        return None, f"Scores log file not found: {scores_log_path}"
    except Exception as e:
        return None, f"Error loading scores log: {str(e)}"


def plot_score_distribution(df, score_column):
    if score_column not in df.columns:
        return None
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df[score_column].dropna(), kde=True, bins=20, color='skyblue', ax=ax)
    if 'BERT' in score_column:
        good_threshold = 0.7
        medium_threshold = 0.5
    else:
        good_threshold = 0.5
        medium_threshold = 0.3
    ax.axvline(x=good_threshold, color='green', linestyle='--', alpha=0.7, label=f'Good (≥{good_threshold})')
    ax.axvline(x=medium_threshold, color='orange', linestyle='--', alpha=0.7, label=f'Medium (≥{medium_threshold})')
    ax.axvspan(good_threshold, 1.0, alpha=0.1, color='green')
    ax.axvspan(medium_threshold, good_threshold, alpha=0.1, color='orange')
    ax.axvspan(0, medium_threshold, alpha=0.1, color='red')
    title_map = {
        'ResponseChunkBERTScore': 'Response-Chunk BERTScore Score Distribution',
        'ResponseAnswerBERTScore': 'Response-Answer BERTScore Score Distribution',
        'ChunkAnswerBERTScore': 'Chunk-Answer BERTScore Score Distribution',
        'ResponseChunkRougeL': 'Response-Chunk ROUGE-L Score Distribution',
        'ResponseAnswerRougeL': 'Response-Answer ROUGE-L Score Distribution',
        'ChunkAnswerRougeL': 'Chunk-Answer ROUGE-L Score Distribution'
    }
    ax.set_title(title_map.get(score_column, f'{score_column} Distribution'), fontsize=14)
    ax.set_xlabel('Score Value', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend()
    mean_score = df[score_column].mean()
    median_score = df[score_column].median()
    stats_text = f'Mean: {mean_score:.4f}\nMedian: {median_score:.4f}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    st.pyplot(fig)
    return fig


def plot_pdf_comparison(df, score_column):
    if 'selected_pdf' not in df.columns or score_column not in df.columns:
        return None
    pdf_stats = df.groupby('selected_pdf')[score_column].agg(['mean', 'count']).reset_index()
    pdf_stats.columns = ['PDF', 'Average Score', 'Query Count']
    pdf_stats = pdf_stats.sort_values('Average Score', ascending=False)
    if len(pdf_stats) > 15:
        pdf_stats = pdf_stats.head(15)
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(pdf_stats['PDF'], pdf_stats['Average Score'], color='skyblue')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                f"n={pdf_stats['Query Count'].iloc[i]}",
                ha='center', va='bottom', fontsize=9)
    if 'BERT' in score_column:
        good_threshold = 0.7
        medium_threshold = 0.5
    else:
        good_threshold = 0.5
        medium_threshold = 0.3
    ax.axhline(y=good_threshold, color='green', linestyle='--', alpha=0.7, label=f'Good (≥{good_threshold})')
    ax.axhline(y=medium_threshold, color='orange', linestyle='--', alpha=0.7, label=f'Medium (≥{medium_threshold})')
    title_map = {
        'ResponseChunkBERTScore': 'Average Response-Chunk BERTScore Score by PDF',
        'ResponseAnswerBERTScore': 'Average Response-Answer BERTScore Score by PDF',
        'ChunkAnswerBERTScore': 'Average Chunk-Answer BERTScore Score by PDF',
        'ResponseChunkRougeL': 'Average Response-Chunk ROUGE-L Score by PDF',
        'ResponseAnswerRougeL': 'Average Response-Answer ROUGE-L Score by PDF',
        'ChunkAnswerRougeL': 'Average Chunk-Answer ROUGE-L Score by PDF'
    }
    ax.set_title(title_map.get(score_column, f'Average {score_column} by PDF'), fontsize=14)
    ax.set_xlabel('PDF', fontsize=12)
    ax.set_ylabel('Average Score', fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_xticklabels(pdf_stats['PDF'], rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    return fig


def plot_model_comparison(df, score_column):
    if 'LLM Model' not in df.columns or score_column not in df.columns:
        return None
    model_stats = df.groupby('LLM Model')[score_column].agg(['mean', 'count']).reset_index()
    model_stats.columns = ['Model', 'Average Score', 'Query Count']
    model_stats = model_stats.sort_values('Average Score', ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(model_stats['Model'], model_stats['Average Score'], color='lightgreen')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                f"n={model_stats['Query Count'].iloc[i]}",
                ha='center', va='bottom', fontsize=9)
    if 'BERTScore' in score_column:
        good_threshold = 0.7
        medium_threshold = 0.5
    else:
        good_threshold = 0.5
        medium_threshold = 0.3
    ax.axhline(y=good_threshold, color='green', linestyle='--', alpha=0.7, label=f'Good (≥{good_threshold})')
    ax.axhline(y=medium_threshold, color='orange', linestyle='--', alpha=0.7, label=f'Medium (≥{medium_threshold})')
    title_map = {
        'ResponseChunkBERTScore': 'Average Response-Chunk BERTScore Score by LLM Model',
        'ResponseAnswerBERTScore': 'Average Response-Answer BERTScore Score by LLM Model',
        'ChunkAnswerBERTScore': 'Average Chunk-Answer BERTScore Score by LLM Model',
        'ResponseChunkRougeL': 'Average Response-Chunk ROUGE-L Score by LLM Model',
        'ResponseAnswerRougeL': 'Average Response-Answer ROUGE-L Score by LLM Model',
        'ChunkAnswerRougeL': 'Average Chunk-Answer ROUGE-L Score by LLM Model'
    }
    ax.set_title(title_map.get(score_column, f'Average {score_column} by LLM Model'), fontsize=14)
    ax.set_xlabel('LLM Model', fontsize=12)
    ax.set_ylabel('Average Score', fontsize=12)
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    return fig


def main():
    st.set_page_config(page_title="Scores Visualization", layout="wide")
    st.markdown(get_css(), unsafe_allow_html=True)
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        font-size: 2.2rem;
        margin-bottom: 20px;
        color: #4169E1;
    }
    .chart-title {
        text-align: center;
        font-size: 1.5rem;
        margin-bottom: 10px;
        color: #333;
    }
    .metrics-row {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .nav-buttons {
        display: flex;
        justify-content: space-between;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">Scores Visualization Dashboard</h1>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Back to Chatbot", key="chatbot_btn", use_container_width=True):
            st.switch_page("main_menu.py")
    with col2:
        if st.button("Go to Score logs", key="kb_btn", use_container_width=True):
            st.switch_page("pages/knowledge_and_scores_viewer.py")
    with col3:
        if st.button("Metrics", key="metrics_btn", use_container_width=True):
            st.switch_page("pages/metrics.py")
    scores_df, error = load_scores_log()
    if error:
        st.error(error)
        st.warning("Please check the file path and try again.")
        return
    st.subheader("Scores Overview")
    bert_tab, rouge_tab = st.tabs(["BERTScore Scores", "ROUGE-L Scores"])
    with bert_tab:
        if 'ResponseChunkBERTScore' in scores_df.columns:
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            with metrics_col1:
                st.metric("Total Entries", len(scores_df))
            with metrics_col2:
                good_pct = len(scores_df[scores_df['ResponseChunkBERTScore'] >= 0.7]) / len(scores_df) * 100
                st.metric("Good BERTScore Scores (≥0.7)", f"{good_pct:.1f}%")
            with metrics_col3:
                medium_pct = len(scores_df[(scores_df['ResponseChunkBERTScore'] >= 0.5) &
                                           (scores_df['ResponseChunkBERTScore'] < 0.7)]) / len(scores_df) * 100
                st.metric("Medium BERTScore Scores (≥0.5)", f"{medium_pct:.1f}%")
            with metrics_col4:
                poor_pct = len(scores_df[scores_df['ResponseChunkBERTScore'] < 0.5]) / len(scores_df) * 100
                st.metric("Poor BERTScore Scores (<0.5)", f"{poor_pct:.1f}%")
        else:
            st.warning("BERTScore score columns not found in the data")
    with rouge_tab:
        if 'ResponseChunkRougeL' in scores_df.columns:
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            with metrics_col1:
                st.metric("Total Entries", len(scores_df))
            with metrics_col2:
                good_pct = len(scores_df[scores_df['ResponseChunkRougeL'] >= 0.5]) / len(scores_df) * 100
                st.metric("Good ROUGE-L Scores (≥0.5)", f"{good_pct:.1f}%")
            with metrics_col3:
                medium_pct = len(scores_df[(scores_df['ResponseChunkRougeL'] >= 0.3) &
                                           (scores_df['ResponseChunkRougeL'] < 0.5)]) / len(scores_df) * 100
                st.metric("Medium ROUGE-L Scores (≥0.3)", f"{medium_pct:.1f}%")
            with metrics_col4:
                poor_pct = len(scores_df[scores_df['ResponseChunkRougeL'] < 0.3]) / len(scores_df) * 100
                st.metric("Poor ROUGE-L Scores (<0.3)", f"{poor_pct:.1f}%")
        else:
            st.warning("ROUGE-L score columns not found in the data")
    st.subheader("Chart Selection")
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        if 'selected_pdf' in scores_df.columns:
            pdf_options = ["All PDFs"] + sorted(scores_df['selected_pdf'].unique().tolist())
            selected_pdf_filter = st.selectbox("Filter by PDF", options=pdf_options)
            if selected_pdf_filter != "All PDFs":
                scores_df = scores_df[scores_df['selected_pdf'] == selected_pdf_filter]
    with filter_col2:
        if 'LLM Model' in scores_df.columns:
            model_options = ["All Models"] + sorted(scores_df['LLM Model'].unique().tolist())
            selected_model_filter = st.selectbox("Filter by LLM Model", options=model_options)
            if selected_model_filter != "All Models":
                scores_df = scores_df[scores_df['LLM Model'] == selected_model_filter]
    score_type = st.radio(
        "Select score type:",
        options=["BERTScore Scores", "ROUGE-L Scores"],
        horizontal=True
    )
    if score_type == "BERTScore Scores":
        score_options = []
        if 'ResponseChunkBERTScore' in scores_df.columns:
            score_options.append("Response-Chunk")
        if 'ResponseAnswerBERTScore' in scores_df.columns:
            score_options.append("Response-Answer")
        if 'ChunkAnswerBERTScore' in scores_df.columns:
            score_options.append("Chunk-Answer")
        selected_score_comparison = st.selectbox(
            "Select BERTScore score comparison:",
            options=score_options
        )
        score_column_map = {
            "Response-Chunk": "ResponseChunkBERTScore",
            "Response-Answer": "ResponseAnswerBERTScore",
            "Chunk-Answer": "ChunkAnswerBERTScore"
        }
        selected_score_column = score_column_map.get(selected_score_comparison)
    else:
        score_options = []
        if 'ResponseChunkRougeL' in scores_df.columns:
            score_options.append("Response-Chunk")
        if 'ResponseAnswerRougeL' in scores_df.columns:
            score_options.append("Response-Answer")
        if 'ChunkAnswerRougeL' in scores_df.columns:
            score_options.append("Chunk-Answer")
        selected_score_comparison = st.selectbox(
            "Select ROUGE-L score comparison:",
            options=score_options
        )
        score_column_map = {
            "Response-Chunk": "ResponseChunkRougeL",
            "Response-Answer": "ResponseAnswerRougeL",
            "Chunk-Answer": "ChunkAnswerRougeL"
        }
        selected_score_column = score_column_map.get(selected_score_comparison)
    chart_types = [
        "Score Distribution Histogram",
        "PDF Comparison Bar Chart",
        "LLM Model Comparison"
    ]
    if 'selected_pdf' not in scores_df.columns:
        chart_types.remove("PDF Comparison Bar Chart")
    if 'LLM Model' not in scores_df.columns:
        chart_types.remove("LLM Model Comparison")
    selected_chart = st.selectbox(
        "Select chart type:",
        options=chart_types
    )
    st.subheader(f"Visualization: {selected_chart} ({score_type})")
    filter_info = []
    if selected_pdf_filter != "All PDFs":
        filter_info.append(f"PDF: {selected_pdf_filter}")
    if 'selected_model_filter' in locals() and selected_model_filter != "All Models":
        filter_info.append(f"Model: {selected_model_filter}")
    if filter_info:
        st.info(f"Filtered by: {', '.join(filter_info)}")
    if selected_chart == "Score Distribution Histogram":
        plot_score_distribution(scores_df, selected_score_column)
    elif selected_chart == "PDF Comparison Bar Chart":
        plot_pdf_comparison(scores_df, selected_score_column)
    elif selected_chart == "LLM Model Comparison":
        plot_model_comparison(scores_df, selected_score_column)
    with st.expander("View Raw Data"):
        st.dataframe(scores_df)
        st.subheader("Dataset Statistics")
        stat_col1, stat_col2 = st.columns(2)
        with stat_col1:
            if 'selected_pdf' in scores_df.columns:
                st.write(f"Number of unique PDFs: {scores_df['selected_pdf'].nunique()}")
            if 'LLM Model' in scores_df.columns:
                st.write(f"Number of LLM models: {scores_df['LLM Model'].nunique()}")
        with stat_col2:
            for col in ['ResponseChunkBERTScore', 'ResponseAnswerBERTScore', 'ChunkAnswerBERTScore',
                        'ResponseChunkRougeL', 'ResponseAnswerRougeL', 'ChunkAnswerRougeL']:
                if col in scores_df.columns:
                    st.write(f"{col} range: {scores_df[col].min():.4f} - {scores_df[col].max():.4f}")


if __name__ == "__main__":
    main()