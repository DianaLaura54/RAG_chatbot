import os
import sys
import streamlit as st


root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from styling.styles import get_css
st.set_page_config(
    page_title="Mode Selection",
    layout="centered"
)

def main():
    st.markdown(get_css(), unsafe_allow_html=True)
    col_title = st.columns([1, 3, 1])
    with col_title[1]:
        st.markdown("<h1 class='header'>Select Your Mode</h1>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    if 'mode' not in st.session_state:
        st.session_state.mode = None
    def set_mode(selected_mode):
        st.session_state.mode = selected_mode
    with col1:
        if st.button("Generate a new quiz", key="normal3", use_container_width=True):
            set_mode("normal3")
    with col2:
        if st.button("Already existing quiz", key="advanced3", use_container_width=True):
            set_mode("advanced3")
    if st.session_state.mode == "normal3":
        gui_path = "pages/quiz_generation.py"
        st.switch_page(gui_path)
    elif st.session_state.mode == "advanced3":
        gui_path = "pages/quiz_selector.py"
        st.switch_page(gui_path)


if __name__ == "__main__":
    main()