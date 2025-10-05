import os
import sys
import streamlit as st

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from styling.styles import get_css

st.set_page_config(
    page_title="StorySage - Mode Selection",
    layout="centered"
)


def main():
    st.markdown(get_css(), unsafe_allow_html=True)
    col_title = st.columns([1, 3, 1])
    with col_title[1]:
        st.markdown("<h1 class='header'>Select Your Mode</h1>", unsafe_allow_html=True)

    if 'mode' not in st.session_state:
        st.session_state.mode = None

    def set_mode(selected_mode):
        st.session_state.mode = selected_mode


    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("Chatbot Mode", key="normal2", use_container_width=True):
            set_mode("normal2")

    with col2:
        if st.button("Quiz Mode", key="advanced2", use_container_width=True):
            set_mode("advanced2")

    with col3:
        if st.button("Validation Dashboard", key="validation", use_container_width=True):
            set_mode("validation")

    with col4:
        try:
            import boto3
            aws_available = True
            button_text = "AWS StorySage"
            help_text = "Cloud-powered document processing with AWS services"
        except ImportError:
            aws_available = False
            button_text = "AWS StorySage"
            help_text = "Install boto3 to enable AWS features: pip install boto3"

        if st.button(button_text, key="aws_mode", use_container_width=True, help=help_text):
            if aws_available:
                set_mode("aws")
            else:
                st.error("Please install boto3 to use AWS features: `pip install boto3`")


    if st.session_state.mode == "normal2":
        st.switch_page("pages/mode_selection_chat.py")
    elif st.session_state.mode == "advanced2":
        st.switch_page("pages/mode_selection_quiz.py")
    elif st.session_state.mode == "validation":
        st.switch_page("pages/validation_dashboard.py")
    elif st.session_state.mode == "aws":
        st.switch_page("pages/aws_mode_selection.py")


if __name__ == "__main__":
    main()
