import os
import sys
import streamlit as st

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from styling.styles import get_css

st.set_page_config(
    page_title="AWS Mode Selection",
    layout="centered"
)


def main():
    st.markdown(get_css(), unsafe_allow_html=True)
    col_title = st.columns([1, 3, 1])
    with col_title[1]:
        st.markdown("<h1 class='header'>AWS StorySage</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #666;</p>",unsafe_allow_html=True)
    if 'aws_mode' not in st.session_state:
        st.session_state.aws_mode = None

    def set_aws_mode(selected_mode):
        st.session_state.aws_mode = selected_mode
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("AWS Configuration", key="aws_config", use_container_width=True):
            set_aws_mode("config")
    with col2:
        if st.button("AWS Document Manager", key="aws_docs", use_container_width=True):
            set_aws_mode("docs")
    with col3:
        if st.button("AWS Chat Mode", key="aws_chat", use_container_width=True):
            set_aws_mode("chat")
    if st.session_state.aws_mode == "config":
        st.switch_page("pages/aws_configuration.py")
    elif st.session_state.aws_mode == "docs":
        st.switch_page("pages/aws_document_manager.py")
    elif st.session_state.aws_mode == "chat":
        st.switch_page("pages/aws_chat_mode.py")
    if st.button("Back to Main Menu", key="back_main"):
        st.switch_page("main_menu.py")


if __name__ == "__main__":
    main()