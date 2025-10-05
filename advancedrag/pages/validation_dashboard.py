import os
import sys
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json


root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from ge_integration import RAGDataValidator, validate_and_process_documents
from processing import get_all_files_in_folder, process_files
from chunking import chunk_documents, chunk_documents_semantic
from embeddings import get_embedding_model, batch_generate_embeddings
from styling.styles import get_css
import numpy as np

st.set_page_config(
    page_title="StorySage Validation Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_validation_history():
    history_file = os.path.join("Contents", "validation_history.json")
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                return json.load(f)
        except:
            return []
    return []



def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    else:
        return obj


def save_validation_result(result):
    history_file = os.path.join("Contents", "validation_history.json")
    history = load_validation_history()
    result['timestamp'] = datetime.now().isoformat()
    result = convert_numpy_types(result)
    history.append(result)
    history = history[-100:]
    os.makedirs(os.path.dirname(history_file), exist_ok=True)
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)


def create_validation_metrics_charts(validation_results):
    charts = {}
    if validation_results:
        status_counts = {'Passed': 0, 'Failed': 0, 'Warnings': 0}
        for stage, results in validation_results.items():
            if results.get('is_valid', False):
                status_counts['Passed'] += 1
            else:
                status_counts['Failed'] += 1
            if results.get('warnings', []):
                status_counts['Warnings'] += 1
        charts['status'] = px.pie(
            values=list(status_counts.values()),
            names=list(status_counts.keys()),
            title="Validation Status Overview",
            color_discrete_map={
                'Passed': '#28a745',
                'Failed': '#dc3545',
                'Warnings': '#ffc107'
            }
        )
    return charts


def display_validation_summary(validation_results):
    if not validation_results:
        st.warning("No validation results available")
        return
    col1, col2, col3, col4 = st.columns(4)
    total_stages = len(validation_results)
    passed_stages = sum(1 for r in validation_results.values() if r.get('is_valid', False))
    total_errors = sum(len(r.get('errors', [])) for r in validation_results.values())
    total_warnings = sum(len(r.get('warnings', [])) for r in validation_results.values())
    with col1:
        st.metric("Total Stages", total_stages)
    with col2:
        st.metric("Passed", passed_stages, delta=f"{passed_stages}/{total_stages}")
    with col3:
        st.metric("Errors", total_errors, delta=None if total_errors == 0 else f"-{total_errors}")
    with col4:
        st.metric("Warnings", total_warnings, delta=None if total_warnings == 0 else f"-{total_warnings}")
    if total_errors == 0:
        st.success(" All validation stages passed!")
    else:
        st.error(f" Validation failed with {total_errors} errors")


def display_detailed_results(validation_results):
    for stage_name, results in validation_results.items():
        with st.expander(f" {stage_name.title().replace('_', ' ')} Results", expanded=False):
            if results.get('is_valid', False):
                st.success(" Validation Passed")
            else:
                st.error(" Validation Failed")
            if results.get('errors', []):
                st.subheader("Errors:")
                for error in results['errors']:
                    st.error(f" {error}")
            if results.get('warnings', []):
                st.subheader("Warnings:")
                for warning in results['warnings']:
                    st.warning(f" {warning}")
            if results.get('statistics', {}):
                st.subheader("Statistics:")
                stats_df = pd.DataFrame([results['statistics']]).T
                stats_df.columns = ['Value']
                st.dataframe(stats_df)


def run_manual_validation():
    st.subheader("Run Manual Validation")
    folder_path = os.path.join("Contents", "books")
    col1, col2 = st.columns([2, 1])
    with col1:
        chunking_method = st.selectbox(
            "Chunking Method:",
            ["standard", "semantic"],
            help="Choose the text chunking method"
        )
    with col2:
        embedding_model = st.selectbox(
            "Embedding Model:",
            ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "multi-qa-MiniLM-L6-cos-v1"],
            help="Choose the embedding model"
        )
    if st.button(" Run Validation", type="primary"):
        with st.spinner("Running comprehensive validation..."):
            try:
                validator = RAGDataValidator("Contents")
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.text("Loading documents...")
                progress_bar.progress(20)
                all_files = get_all_files_in_folder(folder_path)
                if not all_files:
                    st.error("No files found in the books folder")
                    return
                all_documents_with_pages, file_sources = [], []
                for file_path in all_files:
                    file_docs = process_files(file_path)
                    if file_docs:
                        all_documents_with_pages.append(file_docs)
                        file_sources.append(file_path)
                status_text.text("Validating PDF processing...")
                progress_bar.progress(40)
                pdf_validation = validator.validate_pdf_processing_results(
                    all_documents_with_pages, file_sources
                )
                status_text.text("Processing chunks...")
                progress_bar.progress(60)
                #semantic chunking
                if chunking_method == "semantic":
                    chunks, metadata = chunk_documents_semantic(
                        all_documents_with_pages, file_sources, get_embedding_model()
                    )
                    #standard chunking
                else:
                    chunks, metadata = chunk_documents(all_documents_with_pages, file_sources)
                    #validate the chunking with great expectations
                chunking_validation = validator.validate_chunking_results(chunks, metadata)
                #generate the embeddings
                status_text.text("Generating embeddings...")
                progress_bar.progress(80)
                #generate the embeddings
                embeddings = batch_generate_embeddings(chunks)
                #validate them with great expectations
                embeddings_validation = validator.validate_embeddings(embeddings, chunks)
                progress_bar.progress(100)
                status_text.text("Generating report...")
                all_validations = {
                    'pdf_processing': pdf_validation,
                    'chunking': chunking_validation,
                    'embeddings': embeddings_validation
                }
                save_validation_result({
                    'chunking_method': chunking_method,
                    'embedding_model': embedding_model,
                    'results': all_validations
                })
                st.success(" Validation completed!")
                progress_bar.empty()
                status_text.empty()
                #validation results and report
                #report with the validator
                st.session_state.validation_results = all_validations
                st.session_state.validation_report = validator.generate_validation_report(all_validations)
                #summary
                display_validation_summary(all_validations)
                st.subheader(" Detailed Results")
                display_detailed_results(all_validations)
            except Exception as e:
                st.error(f"Validation failed: {str(e)}")
                st.exception(e)


def display_validation_history():
    st.subheader(" Validation History")
    history = load_validation_history()
    if not history:
        st.info("No validation history available. Run a validation first.")
        return
    history_data = []
    for entry in history:
        timestamp = datetime.fromisoformat(entry['timestamp'])
        total_errors = sum(len(r.get('errors', [])) for r in entry['results'].values())
        total_warnings = sum(len(r.get('warnings', [])) for r in entry['results'].values())
        passed_stages = sum(1 for r in entry['results'].values() if r.get('is_valid', False))
        total_stages = len(entry['results'])
        history_data.append({
            'Timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'Chunking Method': entry.get('chunking_method', 'Unknown'),
            'Embedding Model': entry.get('embedding_model', 'Unknown'),
            'Status': 'Passed' if total_errors == 0 else 'Failed',
            'Passed Stages': f"{passed_stages}/{total_stages}",
            'Errors': total_errors,
            'Warnings': total_warnings
        })
    df = pd.DataFrame(history_data)
    st.dataframe(df, use_container_width=True)
    if len(history_data) > 1:
        col1, col2 = st.columns(2)
        with col1:
            status_chart = px.histogram(
                df, x='Status',
                title="Validation Results Over Time",
                color='Status',
                color_discrete_map={'Passed': '#28a745', 'Failed': '#dc3545'}
            )
            st.plotly_chart(status_chart, use_container_width=True)
        with col2:
            if 'Timestamp' in df.columns:
                trend_chart = go.Figure()
                trend_chart.add_trace(go.Scatter(
                    x=df['Timestamp'], y=df['Errors'],
                    mode='lines+markers', name='Errors',
                    line=dict(color='red')
                ))
                trend_chart.add_trace(go.Scatter(
                    x=df['Timestamp'], y=df['Warnings'],
                    mode='lines+markers', name='Warnings',
                    line=dict(color='orange')
                ))
                trend_chart.update_layout(title="Error/Warning Trends")
                st.plotly_chart(trend_chart, use_container_width=True)


def display_validation_report():
    if 'validation_report' in st.session_state:
        st.subheader("Full Validation Report")
        report_text = st.session_state.validation_report
        st.download_button(
            label=" Download Report",
            data=report_text,
            file_name=f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
        st.code(report_text, language=None)


def main():
    st.markdown(get_css(), unsafe_allow_html=True)
    st.title("StorySage Data Validation Dashboard")
    st.markdown("Monitor and validate the chatbot StorySage system's data quality with Great Expectations integration")
    st.sidebar.title(" Validation Tools")
    page = st.sidebar.selectbox(
        "Choose Action:",
        ["Run Validation", "View History", "System Status"]
    )
    if st.sidebar.button(" Back to Main Menu"):
        st.switch_page("main_menu.py")
    if page == "Run Validation":
        run_manual_validation()
        if 'validation_report' in st.session_state:
            st.markdown("---")
            display_validation_report()
    elif page == "View History":
        display_validation_history()
    elif page == "System Status":
        st.subheader(" System Status")
        try:
            from ge_integration import GX_AVAILABLE, RAGDataValidator
            if GX_AVAILABLE:
                st.success(" Great Expectations is available and configured")
                validator = RAGDataValidator()
                st.info(f"Validator initialized successfully")
            else:
                st.warning(" Great Expectations not available - using basic validation")
        except Exception as e:
            st.error(f" Validation system error: {str(e)}")
        books_folder = os.path.join("Contents", "books")
        if os.path.exists(books_folder):
            files = [f for f in os.listdir(books_folder) if f.endswith(('.pdf', '.txt', '.docx'))]
            st.success(f" Found {len(files)} documents in books folder")
            if files:
                st.write("Documents:")
                for file in files[:10]:
                    st.write(f"{file}")
                if len(files) > 10:
                    st.write(f"... and {len(files) - 10} more")
        else:
            st.error(" Books folder not found")



if __name__ == "__main__":
    main()