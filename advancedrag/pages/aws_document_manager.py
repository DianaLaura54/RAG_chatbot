import os
import sys
import streamlit as st
import tempfile

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from styling.styles import get_css
from aws_shared_utils import (
    initialize_aws_session_state,
    check_aws_prerequisites,
    render_navigation,
    initialize_s3_manager,
    get_aws_config
)


def main():
    st.set_page_config(page_title="AWS Document Manager", layout="wide")
    st.markdown(get_css(), unsafe_allow_html=True)
    initialize_aws_session_state()
    st.markdown('<h1 class="main-header">AWS Document Manager</h1>', unsafe_allow_html=True)
    if not check_aws_prerequisites(allow_force_enable=False, require_documents=False):
        return
    render_navigation(current_page='docs')
    st.markdown("---")
    aws_config = get_aws_config()
    s3_manager = initialize_s3_manager(aws_config)
    if s3_manager:
        tab1, tab2, tab3 = st.tabs(["Upload Documents", "Manage Documents", "Process Documents"])
        with tab1:
            render_upload_interface(s3_manager)
        with tab2:
            render_document_list(s3_manager)
        with tab3:
            render_processing_interface(aws_config)
    else:
        st.error("Failed to initialize S3 manager. Please check your AWS configuration.")


def render_upload_interface(s3_manager):
    st.subheader(" Upload Documents to AWS S3")
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        accept_multiple_files=True,
        type=['pdf', 'txt', 'docx', 'csv']
    )
    if uploaded_files:
        st.info(f"Selected {len(uploaded_files)} file(s) for upload")
        if st.button("Upload to S3", type="primary", key="upload_to_s3_btn"):
            upload_files_to_s3(uploaded_files, s3_manager)


def upload_files_to_s3(uploaded_files, s3_manager):
    progress_bar = st.progress(0)
    status_text = st.empty()
    successful_uploads = 0
    failed_uploads = []
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Uploading {uploaded_file.name}...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        try:
            s3_manager.upload_document(tmp_file_path, f"documents/{uploaded_file.name}")
            st.success(f" Uploaded {uploaded_file.name}")
            successful_uploads += 1
        except Exception as e:
            st.error(f" Failed to upload {uploaded_file.name}: {str(e)}")
            failed_uploads.append(uploaded_file.name)
        finally:
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
        progress_bar.progress((i + 1) / len(uploaded_files))
    status_text.text(f"Upload complete! {successful_uploads}/{len(uploaded_files)} files uploaded successfully.")
    if failed_uploads:
        st.warning(f"Failed uploads: {', '.join(failed_uploads)}")


def render_document_list(s3_manager):
    st.subheader(" Documents in AWS S3")
    try:
        with st.spinner("Loading documents from S3..."):
            documents = s3_manager.list_documents("documents/")
        if documents:
            st.success(f"Found {len(documents)} document(s)")
            display_documents_table(documents)
            render_document_actions(documents, s3_manager)
        else:
            st.info("No documents found in S3. Upload some documents first!")
    except Exception as e:
        st.error(f"Error listing documents: {str(e)}")
        st.exception(e)


def display_documents_table(documents):
    try:
        import pandas as pd
        df_data = []
        for doc in documents:
            df_data.append({
                'Filename': doc['filename'],
                'Size (KB)': round(doc['size'] / 1024, 2),
                'Last Modified': doc['last_modified'].strftime('%Y-%m-%d %H:%M:%S'),
                'S3 Key': doc['key']
            })
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True)
    except ImportError:
        st.error("pandas not available for table display")
        for doc in documents:
            st.write(f" {doc['filename']} ({round(doc['size'] / 1024, 2)} KB)")


def render_document_actions(documents, s3_manager):
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button(" Refresh List", key="refresh_doc_list"):
            st.rerun()
    with col2:
        selected_doc = st.selectbox(
            "Select document for preview URL",
            options=[doc['key'] for doc in documents],
            format_func=lambda x: os.path.basename(x),
            key="preview_doc_select"
        )
        if st.button(" Generate Preview URL", key="generate_preview_url"):
            try:
                url = s3_manager.get_presigned_url(selected_doc, expiration=3600)
                st.success(" Preview URL (valid for 1 hour):")
                st.code(url)
            except Exception as e:
                st.error(f"Failed to generate URL: {str(e)}")
    with col3:
        doc_to_delete = st.selectbox(
            "Select document to delete",
            options=[doc['key'] for doc in documents],
            format_func=lambda x: os.path.basename(x),
            key="delete_select"
        )
        if st.button(" Delete Document", type="secondary", key="delete_doc_btn"):
            handle_document_deletion(doc_to_delete, s3_manager)


def handle_document_deletion(doc_key, s3_manager):
    if st.session_state.get('confirm_delete'):
        try:
            if s3_manager.delete_document(doc_key):
                st.success(f"✓ Deleted {os.path.basename(doc_key)}")
                st.rerun()
            else:
                st.error("Failed to delete document")
        except Exception as e:
            st.error(f"Delete failed: {str(e)}")
        st.session_state.confirm_delete = False
    else:
        st.session_state.confirm_delete = True
        st.warning("Click again to confirm deletion")


def render_processing_interface(aws_config):
    st.subheader(" Process Documents with AWS")
    if not aws_config.get('s3_bucket') or not aws_config.get('region'):
        st.error("AWS configuration incomplete. Please configure AWS settings first.")
        st.write("**Current AWS configuration:**")
        st.json(aws_config)
        if st.button("Go to AWS Configuration", key="processing_aws_config"):
            st.switch_page("pages/aws_configuration.py")
        return
    st.success(f" Using S3 bucket: **{aws_config.get('s3_bucket')}** in region **{aws_config.get('region')}**")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Processing Options")
        use_textract = st.checkbox("Use AWS Textract for PDF extraction",
                                   value=aws_config.get('use_textract', False))
        use_comprehend = st.checkbox("Use AWS Comprehend for text analysis",
                                     value=aws_config.get('use_comprehend', False))
        chunking_method = st.selectbox("Chunking Method", ["standard", "semantic"])
    with col2:
        st.markdown("#### Advanced Settings")
        embedding_model = st.selectbox("Embedding Model", [
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "multi-qa-MiniLM-L6-cos-v1"
        ])
        num_results = st.slider("Number of results per search", 3, 10, 5)
    with st.expander("Current AWS Configuration", expanded=False):
        st.json(aws_config)
    if st.button(" Start AWS Processing", type="primary", key="start_aws_processing"):
        run_aws_processing(aws_config, use_textract, use_comprehend, chunking_method, embedding_model)


def run_aws_processing(aws_config, use_textract, use_comprehend, chunking_method, embedding_model):
    progress_container = st.container()
    with progress_container:
        st.info("Step 1: Testing S3 connection...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        documents = test_s3_and_list_documents(aws_config, status_text)
        if not documents:
            return
        progress_bar.progress(0.2)
        enhanced_config = aws_config.copy()
        enhanced_config.update({
            'use_textract': use_textract,
            'use_comprehend': use_comprehend,
            'chunking_method': chunking_method,
            'embedding_model': embedding_model
        })
        st.info("Step 2: Initializing AWS processing...")
        display_processing_info(enhanced_config, len(documents), use_textract, use_comprehend)
        status_text.text("Initializing AWS services...")
        progress_bar.progress(0.3)
        try:
            process_with_aws_rag(enhanced_config, progress_bar, status_text)
        except ImportError as ie:
            st.error(f"AWS integration modules not found: {str(ie)}")
            st.warning("The required AWS integration file 'aws_comprehensive_integration.py' is missing")
            st.info("Falling back to basic processing...")
            basic_processing_fallback(enhanced_config)
        except Exception as aws_error:
            st.error(f"AWS service initialization failed: {str(aws_error)}")
            with st.expander("AWS Error Details", expanded=True):
                st.write(f"**Error:** {str(aws_error)}")
                st.exception(aws_error)
            st.info("Trying basic processing fallback...")
            basic_processing_fallback(enhanced_config)


def test_s3_and_list_documents(aws_config, status_text):
    status_text.text("Testing AWS S3 connection...")
    try:
        import boto3
        s3_client = boto3.client('s3', region_name=aws_config['region'])
        s3_client.list_buckets()
        st.success("AWS credentials work")
        s3_client.head_bucket(Bucket=aws_config['s3_bucket'])
        st.success(f" Bucket '{aws_config['s3_bucket']}' accessible")
        response = s3_client.list_objects_v2(
            Bucket=aws_config['s3_bucket'],
            Prefix='documents/'
        )
        documents = response.get('Contents', [])
        if not documents:
            st.error("No documents found in S3 bucket")
            st.info("Upload documents first using the 'Upload Documents' tab")
            with st.expander("Documents Check Details", expanded=True):
                st.write(f"Checked path: s3://{aws_config['s3_bucket']}/documents/")
                st.write("No files found in this location")
            return None
        st.success(f" Found {len(documents)} documents ready for processing")
        with st.expander("Documents Found", expanded=False):
            for i, doc in enumerate(documents[:5]):
                filename = doc['Key'].split('/')[-1]
                size_kb = doc['Size'] / 1024
                st.write(f"  {i + 1}. {filename} ({size_kb:.1f} KB)")
            if len(documents) > 5:
                st.write(f"  ... and {len(documents) - 5} more documents")
        return documents
    except Exception as e:
        st.error(f"S3 connection failed: {str(e)}")
        st.error("Cannot proceed with document processing")
        return None


def display_processing_info(config, doc_count, use_textract, use_comprehend):
    st.info(f"Using S3 bucket: {config.get('s3_bucket')}")
    st.info(f"Using AWS region: {config.get('region')}")
    st.info(f"Processing {doc_count} documents with:")
    st.info(f"  - Textract: {'Yes' if use_textract else 'No'}")
    st.info(f"  - Comprehend: {'Yes' if use_comprehend else 'No'}")


def process_with_aws_rag(enhanced_config, progress_bar, status_text):
    from aws_comprehensive_integration import ComprehensiveAWSRAGSystem
    st.success("AWS integration module found")
    aws_rag_system = ComprehensiveAWSRAGSystem(enhanced_config)
    st.success(" AWS RAG system initialized")
    progress_bar.progress(0.5)
    status_text.text("Processing documents with AWS services...")
    st.info("Step 3: Processing documents (this may take several minutes)...")
    processing_result = aws_rag_system.process_documents_with_aws()
    progress_bar.progress(0.8)
    st.session_state.aws_processing_result = processing_result
    st.session_state.aws_documents_processed = True
    progress_bar.progress(1.0)
    status_text.text("Processing complete!")
    st.success(f"""
    ✓ AWS Processing Complete!
    - Total documents: {processing_result['total_documents']}
    - Successfully processed: {processing_result['processed_documents']}
    - Documents ready for search: {len(processing_result['documents_with_pages'])}
    """)
    st.info("You can now use the AWS Chat Mode to query your documents!")
    with st.expander("Processing Details", expanded=False):
        st.json(processing_result)


def basic_processing_fallback(config):
    st.info("Using basic document processing as AWS integration is not fully available.")
    st.session_state.aws_processing_result = {
        'total_documents': 0,
        'processed_documents': 0,
        'documents_with_pages': [],
        'fallback_mode': True
    }
    st.session_state.aws_documents_processed = True
    st.warning(
        "Limited functionality available. For full AWS features, ensure all AWS integration files are properly installed.")


if __name__ == "__main__":
    main()