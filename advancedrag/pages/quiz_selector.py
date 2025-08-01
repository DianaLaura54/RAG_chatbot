import sys

import streamlit as st
import pandas as pd
import os
import random


root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from styling.styles import get_css


if 'quiz_started' not in st.session_state:
    st.session_state.quiz_started = False

if 'quiz_completed' not in st.session_state:
    st.session_state.quiz_completed = False

if 'current_question' not in st.session_state:
    st.session_state.current_question = 0

if 'questions' not in st.session_state:
    st.session_state.questions = []

if 'score' not in st.session_state:
    st.session_state.score = 0

if 'user_answers' not in st.session_state:
    st.session_state.user_answers = []

if 'selected_quiz_path' not in st.session_state:
    st.session_state.selected_quiz_path = None


def get_available_quizzes():
    quiz_folders = [
        "advancedrag/pages/quiz_outputs",
        "../advancedrag/pages/quiz_outputs",
        "../../advancedrag/pages/quiz_outputs",
        "./advancedrag/pages/quiz_outputs",
        "advancedrag/quiz_outputs",
        "../advancedrag/quiz_outputs",
        "../../advancedrag/quiz_outputs",
        "./advancedrag/quiz_outputs",
        "quiz_outputs",
        "../quiz_outputs",
        "./quiz_outputs"
    ]
    quiz_files = []
    found_folder = None
    for folder in quiz_folders:
        if os.path.exists(folder) and os.path.isdir(folder):
            found_folder = folder
            for file in os.listdir(folder):
                if file.endswith('.csv'):
                    quiz_files.append({
                        'name': file,
                        'path': os.path.join(folder, file)
                    })
            if quiz_files:
                break
    return quiz_files, found_folder


def load_quiz_from_csv(file_path):
    try:
        df = pd.read_csv(file_path,   encoding='utf-8-sig',sep=';',on_bad_lines='skip',low_memory=False)
        required_cols = ['question', 'correct_answer']
        alt_cols = ['distractor1', 'distractor2', 'distractor_1', 'distractor_2', 'alternative1', 'alternative2']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"CSV file missing required columns: {', '.join(missing_cols)}")
            return []
        used_alt_cols = [col for col in alt_cols if col in df.columns]
        questions = []
        for _, row in df.iterrows():
            alternatives = []
            for col in used_alt_cols:
                if pd.notnull(row.get(col)) and str(row.get(col)).strip():
                    alternatives.append(str(row[col]))
            question = {
                'question': row['question'],
                'correct_answer': row['correct_answer'],
                'alternatives': alternatives
            }
            questions.append(question)
        return questions
    except Exception as e:
        st.error(f"Error loading quiz: {str(e)}")
        return []



def display_quiz_selection():
    st.markdown("<h1 style='color:#96053e;'>Quiz Selector</h1>", unsafe_allow_html=True)
    quizzes, folder_path = get_available_quizzes()
    if not quizzes:
        st.warning("No quiz files found in quiz_outputs folder.")
        return
    quiz_names = [q['name'] for q in quizzes]
    selected_quiz = st.selectbox("Select a quiz to take:", quiz_names)
    selected_path = next((q['path'] for q in quizzes if q['name'] == selected_quiz), None)
    if selected_path:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start Quiz"):
                questions = load_quiz_from_csv(selected_path)
                if questions:
                    st.session_state.questions = questions
                    st.session_state.quiz_started = True
                    st.session_state.quiz_completed = False
                    st.session_state.current_question = 0
                    st.session_state.score = 0
                    st.session_state.user_answers = [""] * len(questions)
                    st.session_state.selected_quiz_path = selected_path
                    st.rerun()
                else:
                    st.error("Failed to load quiz questions. Please select another quiz.")


def display_question(question, question_number, total_questions):
    st.markdown(f"<h2 style='color: #96053e; font-weight: bold;'>Question {question_number} of {total_questions}</h2>",unsafe_allow_html=True)
    st.markdown(f"<p style='font-size: 18px; font-weight: 600; margin: 15px 0;'>{question['question']}</p>",unsafe_allow_html=True)
    options = [question['correct_answer']] + question['alternatives']
    random.seed(question_number)
    random.shuffle(options)
    random.seed(None)
    selected_option = st.radio("Select your answer:", options, key=f"q_{question_number}")
    st.session_state.user_answers[question_number - 1] = selected_option
    col1, col2 = st.columns(2)
    with col1:
        if question_number > 1:
            if st.button("Previous"):
                st.session_state.current_question -= 1
                st.rerun()
    with col2:
        if question_number < total_questions:
            if st.button("Next"):
                st.session_state.current_question += 1
                st.rerun()
        else:
            if st.button("Submit Quiz"):
                st.session_state.score = 0
                for i, q in enumerate(st.session_state.questions):
                    if i < len(st.session_state.user_answers) and st.session_state.user_answers[i] == q[
                        'correct_answer']:
                        st.session_state.score += 1
                st.session_state.quiz_completed = True
                st.session_state.quiz_started = False
                st.rerun()


def display_results():
    st.title("Quiz Results")
    score = st.session_state.score
    total = len(st.session_state.questions)
    if total > 0:
        percentage = (score / total) * 100
        col1, col2 = st.columns([3, 1])
        with col1:
            st.header(f"Your Score: {score}/{total} ({percentage:.1f}%)")
            st.subheader("Point Summary")
            for i, q in enumerate(st.session_state.questions):
                user_answer = st.session_state.user_answers[i] if i < len(st.session_state.user_answers) else None
                is_correct = user_answer == q['correct_answer'] if user_answer else False
        if percentage >= 80:
            st.success("Excellent! You really know this material well.")
            st.balloons()
        elif percentage >= 60:
            st.info("Good job! You have a solid understanding of the material.")
        else:
            st.warning("You might want to review the material some more.")
        st.subheader("Question Details")
        for i, q in enumerate(st.session_state.questions):
            user_answer = st.session_state.user_answers[i] if i < len(st.session_state.user_answers) else None
            is_correct = user_answer == q['correct_answer'] if user_answer else False
            with st.expander(f"Question {i + 1}"):
                st.write(f"**Question:** {q['question']}")
                st.write(f"**Correct answer:** {q['correct_answer']}")
                if user_answer:
                    status = "Correct " if is_correct else "Incorrect "
                    st.write(f"**Your answer:** {user_answer} ({status})")
                st.write("**All possible answers:**")
                st.write(f"- {q['correct_answer']} (Correct)")
                for alt in q['alternatives']:
                    st.write(f"- {alt}")
    else:
        st.warning("No questions were found in the quiz.")
    if st.button("Choose Another Quiz"):
        st.session_state.quiz_started = False
        st.session_state.quiz_completed = False
        st.session_state.questions = []
        st.session_state.current_question = 0
        st.session_state.score = 0
        st.session_state.user_answers = []
        st.session_state.selected_quiz_path = None
        st.rerun()


def main():
    if st.session_state.quiz_completed:
        display_results()
    elif st.session_state.quiz_started:
        current_idx = st.session_state.current_question
        questions = st.session_state.questions
        if questions and 0 <= current_idx < len(questions):
            display_question(questions[current_idx], current_idx + 1, len(questions))
        else:
            st.error("Question index out of range. Returning to quiz selection.")
            st.session_state.quiz_started = False
            st.rerun()
    else:
        display_quiz_selection()


if __name__ == "__main__":
    st.markdown(get_css(), unsafe_allow_html=True)
    main()