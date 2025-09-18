
import streamlit as st
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional


def show_validation_status_widget():
    try:
        if 'validation_results' in st.session_state:
            results = st.session_state.validation_results
            timestamp = st.session_state.get('validation_timestamp', datetime.now())

            total_errors = sum(len(r.get('errors', [])) for r in results.values())
            total_warnings = sum(len(r.get('warnings', [])) for r in results.values())

            if total_errors == 0:
                status_color = "#28a745"
                status_text = "System Validated"

            else:
                status_color = "#dc3545"
                status_text = "Validation Issues"


            st.markdown(f"""
            <div style="padding: 8px; border-radius: 5px; background-color: {status_color}20; 
                        border-left: 4px solid {status_color}; margin: 10px 0;">
                <small style="color: {status_color};">
                 {status_text} | Errors: {total_errors} | Warnings: {total_warnings}
                    <br>Last validated: {timestamp.strftime('%H:%M:%S')}
                </small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info(" Run validation to check system status")

    except Exception as e:
        st.error(f"Error displaying validation status: {str(e)}")


def add_validation_sidebar():

    with st.sidebar:
        st.markdown("---")
        st.markdown("### Data Validation")

        if st.button(" Quick Validate", key="quick_validate"):
            run_quick_validation()

        if st.button(" Full Dashboard", key="validation_dashboard"):
            st.switch_page("pages/validation_dashboard.py")

        show_last_validation_status()


def show_last_validation_status():

    try:
        history_file = os.path.join("Contents", "validation_history.json")
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)

            if history:
                latest = history[-1]
                timestamp = datetime.fromisoformat(latest['timestamp'])
                time_ago = datetime.now() - timestamp

                if time_ago.days == 0:
                    if time_ago.seconds < 3600:
                        time_str = f"{time_ago.seconds // 60}m ago"
                    else:
                        time_str = f"{time_ago.seconds // 3600}h ago"
                else:
                    time_str = f"{time_ago.days}d ago"

                results = latest.get('results', {})
                total_errors = sum(len(r.get('errors', [])) for r in results.values())

                if total_errors == 0:
                    st.success(f" Validated {time_str}")
                else:
                    st.error(f" {total_errors} errors {time_str}")
            else:
                st.info("No validation history")
        else:
            st.info("No validation history")

    except Exception as e:
        st.warning(f"Error loading validation status: {str(e)}")


def run_quick_validation():

    with st.spinner("Running quick validation..."):
        try:
            from ge_integration import RAGDataValidator
            from processing import get_all_files_in_folder, process_files

            validator = RAGDataValidator("Contents")
            folder_path = os.path.join("Contents", "books")
            all_files = get_all_files_in_folder(folder_path)

            if not all_files:
                st.error(" No documents found")
                return

            processed_count = 0
            for file_path in all_files[:3]:
                file_docs = process_files(file_path)
                if file_docs:
                    processed_count += 1

            if processed_count > 0:
                st.success(f" Quick validation passed - {processed_count} files processed")
            else:
                st.error(" Quick validation failed - no files could be processed")

        except Exception as e:
            st.error(f" Quick validation failed: {str(e)}")


def validate_during_processing(chunks, metadata, embeddings=None):

    try:
        from ge_integration import RAGDataValidator

        validator = RAGDataValidator("Contents")
        chunking_validation = validator.validate_chunking_results(chunks, metadata)

        results = {'chunking': chunking_validation}

        if embeddings is not None:
            embeddings_validation = validator.validate_embeddings(embeddings, chunks)
            results['embeddings'] = embeddings_validation

        st.session_state.validation_results = results
        st.session_state.validation_timestamp = datetime.now()

        total_errors = sum(len(r.get('errors', [])) for r in results.values())
        total_warnings = sum(len(r.get('warnings', [])) for r in results.values())

        if total_errors == 0:
            st.success(f" Data validation passed (Warnings: {total_warnings})")
        else:
            st.error(f" Data validation failed - {total_errors} errors, {total_warnings} warnings")

        return results

    except Exception as e:
        st.warning(f"Validation error: {str(e)}")
        return None


def add_validation_alerts():
    if 'validation_results' in st.session_state:
        results = st.session_state.validation_results

        total_errors = sum(len(r.get('errors', [])) for r in results.values())
        critical_warnings = []

        for stage, result in results.items():
            for warning in result.get('warnings', []):
                if any(keyword in warning.lower() for keyword in ['critical', 'failed', 'invalid', 'corrupted']):
                    critical_warnings.append(f"{stage}: {warning}")

        if total_errors > 0:
            st.error(f" Data validation failed with {total_errors} errors. Check validation dashboard for details.")

        if critical_warnings:
            with st.expander(" Critical Validation Warnings", expanded=False):
                for warning in critical_warnings:
                    st.warning(warning)