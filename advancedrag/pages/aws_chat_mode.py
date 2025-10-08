import os
import sys
import streamlit as st

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from styling.styles import get_css
from aws_shared_utils import (
    initialize_aws_session_state,
    check_aws_prerequisites,
    render_navigation,
    get_aws_config,
    is_valid_input
)


def main():
    st.set_page_config(page_title="AWS Chat Mode", layout="wide", initial_sidebar_state="collapsed")
    st.markdown(get_css(), unsafe_allow_html=True)
    initialize_aws_session_state()
    st.markdown('<h1 class="main-header">StorySage AWS Edition</h1>', unsafe_allow_html=True)
    if not check_aws_prerequisites(allow_force_enable=True, require_documents=True):
        return
    render_navigation(current_page='chat')
    render_top_controls()
    if st.button("DEBUG: Check Session State"):
        show_session_debug()
    render_chat_interface()
    render_input_area()


def show_session_debug():
    processing_result = st.session_state.get('aws_processing_result')
    if processing_result:
        st.write(" Documents in session state:")
        st.json({
            'total_documents': processing_result.get('total_documents'),
            'processed_documents': processing_result.get('processed_documents'),
            'documents_available': len(processing_result.get('documents_with_pages', []))
        })
    else:
        st.error("✗ No processing result in session state")
        st.write("Go to Document Manager → Process Documents to fix this")


def render_top_controls():
    top_cols = st.columns([3, 1])
    with top_cols[0]:
        st.markdown('<div class="generate-button" style="margin-bottom: 15px;">', unsafe_allow_html=True)
        if st.button("Generate Random Question", key="aws_generate_question_btn"):
            random_questions = [
                "What is the main theme of the stories?",
                "Summarize the key points from the documents",
                "What are the most important findings?",
                "Can you explain the main concepts discussed?",
                "What patterns do you see in the data?"
            ]
            import random
            st.session_state.aws_user_question_input = random.choice(random_questions)
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with top_cols[1]:
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Clear Chat", key="aws_clear_chat_btn"):
                st.session_state.aws_chat_history = []
                st.session_state.aws_tts_audio = {}
                st.session_state.aws_search_results = None
                st.session_state.aws_last_response = None
                st.success("Chat cleared!")
                st.rerun()
        with col2:
            if st.button("System Check", key="aws_diagnostic_btn"):
                st.session_state.aws_user_question_input = "diagnostic"
                st.rerun()
        with col3:
            if st.button("View Logs", key="aws_logs_btn"):
                try:
                    st.switch_page("pages/knowledge_and_scores_viewer.py")
                except:
                    st.info("Logs viewer not available")


def render_chat_interface():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    chat_history = st.session_state.get('aws_chat_history', [])
    if not chat_history:
        st.markdown('<div class="empty-chat">Ask a question about your AWS-processed documents</div>',unsafe_allow_html=True)
    else:
        for i, (sender, message) in enumerate(chat_history):
            if sender == 'You':
                st.markdown(f'<div class="user-message">You: {message}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-message">Bot: {message}</div>', unsafe_allow_html=True)
                tts_key = f"aws_tts_btn_{i}"
                if st.button("Listen to Response", key=tts_key):
                    generate_tts_audio(message, i)
                if i in st.session_state.get('aws_tts_audio', {}):
                    st.audio(st.session_state.aws_tts_audio[i], format='audio/mp3')
    st.markdown('</div>', unsafe_allow_html=True)


def generate_tts_audio(message, index):
    try:
        from audio import generate_audio_from_text, clean_text_for_audio
        clean_text = clean_text_for_audio(message)
        with st.spinner("Generating audio..."):
            audio_buffer = generate_audio_from_text(clean_text)
            if audio_buffer:
                st.session_state.aws_tts_audio[index] = audio_buffer
                st.rerun()
    except ImportError:
        st.warning("Audio functionality not available")


def render_input_area():
    st.markdown('<div class="chat-input-container" style="margin-top: 10px;">', unsafe_allow_html=True)
    input_col, button_col = st.columns([6, 1])
    with input_col:
        user_input = st.text_input(
            "User Question",
            key="aws_user_question_input",
            placeholder="Ask about your documents processed with AWS...",
            label_visibility="hidden"
        )
    with button_col:
        st.markdown('<div class="send-button">', unsafe_allow_html=True)
        send_button = st.button("Send", key="aws_send_button")
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    if send_button and user_input.strip():
        if is_diagnostic_command(user_input):
            process_aws_query(user_input)
        elif is_valid_input(user_input):
            process_aws_query(user_input)
        else:
            st.warning("Please enter a valid question (at least 3 characters with alphanumeric content).")


def is_diagnostic_command(user_input):
    diagnostic_commands = ['diagnostic', 'status', 'debug', 'check system', 'system status']
    return user_input.lower().strip() in diagnostic_commands


def process_aws_query(user_input):
    original_user_input = user_input
    with st.spinner("Processing your question with AWS services..."):
        try:
            aws_config = get_aws_config()
            if not aws_config.get('s3_bucket') or not aws_config.get('region'):
                add_to_chat(original_user_input,"AWS configuration is incomplete. Please check your S3 bucket and region settings.")
                st.rerun()
                return
            if is_diagnostic_command(user_input):
                response = handle_diagnostic(aws_config)
            else:
                response = handle_regular_query(user_input, aws_config)
            add_to_chat(original_user_input, response)
            st.session_state.aws_last_response = response
        except Exception as e:
            error_response = f"Sorry, I encountered an error while processing your question: {str(e)}"
            add_to_chat(original_user_input, error_response)
    st.rerun()


def add_to_chat(user_message, bot_response):
    st.session_state.aws_chat_history.append(('You', user_message))
    st.session_state.aws_chat_history.append(('Bot', bot_response))


def handle_diagnostic(aws_config):
    try:
        from aws_comprehensive_integration import ComprehensiveAWSRAGSystem
        aws_rag_system = ComprehensiveAWSRAGSystem(aws_config)
        if st.session_state.get('aws_processing_result'):
            processing_result = st.session_state.aws_processing_result
            aws_rag_system.documents_with_pages = processing_result.get('documents_with_pages', [])
            aws_rag_system.file_sources = processing_result.get('file_sources', [])
        return generate_diagnostic_response(aws_rag_system, aws_config)
    except Exception as e:
        return f"Diagnostic failed: {str(e)}"


def handle_regular_query(user_input, aws_config):
    try:
        from aws_comprehensive_integration import ComprehensiveAWSRAGSystem
        aws_rag_system = ComprehensiveAWSRAGSystem(aws_config)
        if st.session_state.get('aws_processing_result'):
            processing_result = st.session_state.aws_processing_result
            aws_rag_system.documents_with_pages = processing_result.get('documents_with_pages', [])
            aws_rag_system.file_sources = processing_result.get('file_sources', [])
        search_capable = any(hasattr(aws_rag_system, method) for method in [
            'enhanced_search_with_aws', 'search_documents', 'search', 'query_documents', 'query'
        ])
        if search_capable:
            return generate_aws_enhanced_response(user_input, aws_config, aws_rag_system)
        else:
            return generate_basic_rag_response(user_input, aws_config, aws_rag_system)
    except ImportError:
        return generate_basic_response(user_input, aws_config)
    except Exception as rag_error:
        return generate_error_response(user_input, aws_config, str(rag_error))


def generate_diagnostic_response(aws_rag_system, aws_config):
    diagnostic_info = {
        'documents_found': 0,
        'search_index_status': 'Unknown',
        'sample_content': [],
        'errors': [],
        'processing_status': 'Unknown'
    }
    try:
        if hasattr(aws_rag_system, 's3_manager'):
            s3_manager = aws_rag_system.s3_manager
            try:
                documents = s3_manager.list_documents("documents/")
                diagnostic_info['documents_found'] = len(documents) if documents else 0
                if documents:
                    diagnostic_info['document_list'] = [doc.get('filename', 'Unknown') for doc in documents[:5]]
            except Exception as e:
                diagnostic_info['errors'].append(f"S3 listing error: {str(e)}")
        documents_with_pages = None
        processing_details = []
        possible_attributes = [
            'documents_with_pages', 'processed_documents', '_processed_documents',
            'document_pages', 'indexed_documents', 'processed_content'
        ]
        for attr in possible_attributes:
            if hasattr(aws_rag_system, attr):
                try:
                    docs = getattr(aws_rag_system, attr)
                    if docs:
                        documents_with_pages = docs
                        processing_details.append(f"Found {len(docs)} docs in '{attr}'")
                        diagnostic_info['processing_status'] = f'Processed ({len(docs)} docs via {attr})'
                        break
                except Exception as e:
                    processing_details.append(f"Error accessing '{attr}': {str(e)}")
        if not documents_with_pages and st.session_state.get('aws_processing_result'):
            result = st.session_state.aws_processing_result
            if result and 'documents_with_pages' in result:
                documents_with_pages = result['documents_with_pages']
                processing_details.append(f"Found {len(documents_with_pages)} docs in session state")
                diagnostic_info['processing_status'] = f'Processed ({len(documents_with_pages)} docs in session)'
        if processing_details:
            diagnostic_info['processing_details'] = processing_details
        if not documents_with_pages:
            diagnostic_info['processing_status'] = 'No processed documents found - may need processing'
        if documents_with_pages:
            test_queries = ['the', 'and', 'is', 'document', 'text']
            search_working = False
            for query in test_queries:
                try:
                    results = aws_rag_system.enhanced_search_with_aws(query, documents_with_pages)
                    if results and len(results) > 0:
                        search_working = True
                        diagnostic_info['search_index_status'] = 'Working - Found results'
                        diagnostic_info['sample_content'] = results[:2]
                        break
                except Exception as e:
                    diagnostic_info['errors'].append(f"Search test failed for '{query}': {str(e)}")
            if not search_working:
                diagnostic_info['search_index_status'] = 'No results for common terms'
        else:
            diagnostic_info['search_index_status'] = 'Cannot test - no processed documents'
    except Exception as e:
        diagnostic_info['errors'].append(f"General diagnostic error: {str(e)}")
    response_parts = [
        "**AWS System Diagnostic Report**\n",
        "**Document Status:**",
        f"- Documents in S3: {diagnostic_info['documents_found']}",
        f"- Processing Status: {diagnostic_info['processing_status']}",
        f"- Search Index: {diagnostic_info['search_index_status']}\n"
    ]
    if diagnostic_info.get('document_list'):
        response_parts.append("**Documents Found:**")
        for doc in diagnostic_info['document_list']:
            response_parts.append(f"- {doc}")
        response_parts.append("")
    if diagnostic_info.get('processing_details'):
        response_parts.append("**Processing Investigation:**")
        for detail in diagnostic_info['processing_details']:
            response_parts.append(f"- {detail}")
        response_parts.append("")
    if diagnostic_info.get('sample_content'):
        response_parts.append("**Search is working! Sample results:**")
        for i, result in enumerate(diagnostic_info['sample_content'], 1):
            content = str(result)[:100] + "..." if len(str(result)) > 100 else str(result)
            response_parts.append(f"{i}. {content}")
        response_parts.append("")
    if diagnostic_info['errors']:
        response_parts.append("**Issues Found:**")
        for error in diagnostic_info['errors']:
            response_parts.append(f"- {error}")
        response_parts.append("")
    response_parts.append("**Recommendations:**")
    if diagnostic_info['documents_found'] == 0:
        response_parts.extend([
            "1. Upload documents through the Document Manager",
            "2. Ensure documents are saved to the S3 bucket"
        ])
    elif 'No processed documents' in diagnostic_info['processing_status']:
        response_parts.extend([
            "1. **PRIORITY: Process your documents in Document Manager**",
            "2. Go to Document Manager → 'Process Documents' tab",
            "3. Click 'Start AWS Processing' and wait for completion",
            "4. This will enable search functionality"
        ])
    elif diagnostic_info['search_index_status'] == 'No results for common terms':
        response_parts.extend([
            "1. Documents are processed but search may need optimization",
            "2. Try more specific search terms related to your content",
            "3. Consider re-processing if issues persist"
        ])
    else:
        response_parts.extend([
            "1. System appears healthy - try searching your documents!",
            "2. Use keywords that match your document content"
        ])
    response_parts.extend([
        "\n**AWS Configuration:**",
        f"- S3 Bucket: {aws_config.get('s3_bucket')}",
        f"- Region: {aws_config.get('region')}",
        f"- Textract: {'Yes' if aws_config.get('use_textract') else 'No'}",
        f"- Comprehend: {'Yes' if aws_config.get('use_comprehend') else 'No'}"
    ])
    return "\n".join(response_parts)


def generate_aws_enhanced_response(user_input, aws_config, aws_rag_system):
    try:
        documents_with_pages = getattr(aws_rag_system, 'documents_with_pages', [])
        if not documents_with_pages:
            return "**Need to Re-process Documents**"
        search_results = aws_rag_system._perform_search(user_input, documents_with_pages)
        if search_results and len(search_results) > 0:
            try:
                import requests
                context_parts = []
                for result in search_results[:3]:
                    content = result.get('content', result.get('text', ''))
                    source = result.get('filename', result.get('source', 'Document'))
                    context_parts.append(f"From {source}: {content}")
                context = "\n\n".join(context_parts)
                prompt = f"""Based on the following information from documents, answer the user's question directly and comprehensively.
Context from documents:
{context}
User's question: {user_input}
Please provide a direct, helpful answer based on the information above. If the context doesn't contain enough information to answer the question, say so clearly."""
                response = requests.post(
                    'http://localhost:11434/api/generate',
                    json={"model": "llama3", "prompt": prompt, "stream": False},
                    headers={'Content-Type': 'application/json'},
                    timeout=30
                )
                if response.status_code == 200:
                    ai_answer = response.json().get('response', '').strip()
                    if ai_answer:
                        sources = ', '.join([r.get('filename', 'Document') for r in search_results[:3]])
                        return f"**Answer:**\n{ai_answer}\n\n**Sources:** {sources}"
            except Exception:
                pass
            response_parts = [
                f"**Search Results for: '{user_input}'**\n",
                f"Found {len(search_results)} relevant results:\n"
            ]
            for i, result in enumerate(search_results[:3], 1):
                content = result.get('content', result.get('text', str(result)))
                source = result.get('filename', result.get('source', f'Document {i}'))
                score = result.get('score', 'N/A')
                if len(str(content)) > 300:
                    content = str(content)[:300] + "..."
                response_parts.append(f"**{i}. {source}** (Score: {score})")
                response_parts.append(f"{content}\n")
            return "\n".join(response_parts)
        else:
            return f"No relevant information found for '{user_input}'"
    except Exception as e:
        return f"Search Error: {str(e)}"


def generate_basic_rag_response(user_input, aws_config, aws_rag_system):
    available_methods = [method for method in dir(aws_rag_system) if not method.startswith('_')]
    return f"""**AWS RAG System Response**
Your question: "{user_input}"
The AWS RAG system is initialized but search functionality may need to be set up.
**Available Methods:** {', '.join(available_methods[:10])}
**Next Steps:**
1. Ensure documents have been processed and indexed
2. Check that search methods are properly implemented
3. Try re-processing documents through the Document Manager
"""


def generate_error_response(user_input, aws_config, error_details):
    return f"""**AWS Integration Issue**
Your question: "{user_input}"
**Error:** {error_details}
**Troubleshooting:**
1. Check AWS integration files are installed
2. Verify AWS credentials are configured
3. Ensure documents have been processed
4. Try restarting the application
"""


def generate_basic_response(user_input, aws_config):
    return f"""**StorySage Response**
Your question: "{user_input}"
AWS integration is configured but some components may not be fully available.
**AWS Configuration:**
- S3 Bucket: {aws_config.get('s3_bucket', 'Not configured')}
- Region: {aws_config.get('region', 'Not configured')}
**Next Steps:**
Return to the Document Manager to process your documents with AWS services.
"""


if __name__ == "__main__":
    main()