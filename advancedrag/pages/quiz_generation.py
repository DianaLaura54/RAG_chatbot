import pandas as pd
import os
import random
import requests
import sys
import streamlit as st


root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from styling.styles import get_css


if 'quiz_questions' not in st.session_state:
    st.session_state.quiz_questions = []

if 'current_question_index' not in st.session_state:
    st.session_state.current_question_index = 0

if 'score' not in st.session_state:
    st.session_state.score = 0

if 'quiz_in_progress' not in st.session_state:
    st.session_state.quiz_in_progress = False


if 'quiz_completed' not in st.session_state:
    st.session_state.quiz_completed = False

if 'user_answers' not in st.session_state:
    st.session_state.user_answers = []

if 'saved_csv_path' not in st.session_state:
    st.session_state.saved_csv_path = None


def query_llama3(prompt, user_input, relevant_chunks, correct_answer=""):
    url = 'http://localhost:11434/api/generate'
    relevant_texts = [chunk["text"] for chunk in relevant_chunks if isinstance(chunk, dict) and "text" in chunk] \
        if all(isinstance(chunk, dict) and "text" in chunk for chunk in relevant_chunks) else relevant_chunks
    try:
        formatted_prompt = prompt.format(
            user_input=user_input,
            correct_answer=correct_answer,
            relevant_documents="\n".join(relevant_texts))
    except KeyError as e:
        st.error(f"Prompt formatting error: missing placeholder {e}")
        return None
    data = {
        "model": "llama3:8b",
        "prompt": formatted_prompt,
        "stream": False}
    try:
        response = requests.post(url, json=data, headers={'Content-Type': 'application/json'})
        if response.status_code == 200:
            return response.json().get('response', '')
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"API connection error: {str(e)}")
        return None


def generate_llm_alternatives(question, correct_answer):
    try:
        # template to prompt the llm for generating alternative answers
        prompt_template = (
            "Hey there! I'll help you create multiple-choice quiz options. Based on this question and its correct answer, "
            "I'll generate two alternative answers that are incorrect but plausible.\n\n"
            "Question: {user_input}\n"
            "Correct Answer: {correct_answer}\n\n"
            "The alternatives should be related to the same topic, have similar length, and be plausible but incorrect.\n\n"
            "Please format your response as a list, with each alternative on a new line starting with '- '.")
        # query llama3
        response = query_llama3(
            prompt_template,
            user_input=question,
            relevant_chunks=[],
            correct_answer=correct_answer)
        # if the response is empty or none, return default incorrect answers
        if response is None or not response.strip():
            return [
                f"A common misconception about {question.split()[0] if len(question.split()) > 0 else 'this topic'}",
                f"An incorrect but related approach to {question.split()[-1] if len(question.split()) > 0 else 'this concept'}"]
        alternatives = []
        # iterate through the response lines and extract incorrect alternatives
        for line in response.strip().split('\n'):
            line = line.strip()
            # check if the line starts with "- " or "* " to capture llm's formatted alternatives
            if line.startswith(('- ', '* ')):
                alt_text = line[2:].strip()
                if alt_text and alt_text != correct_answer:  # ensure it's not the correct answer
                    alternatives.append(alt_text)
            # handle numbered responses (e.g., 1. alternative answer)
            elif line.startswith(('1.', '2.', '3.')):
                parts = line.split('.', 1)
                if len(parts) > 1:
                    alt_text = parts[1].strip()
                    if alt_text and alt_text != correct_answer:  # ensure it's not the correct answer
                        alternatives.append(alt_text)
        # if there are less than two alternatives, try to extract more alternatives from the response
        if len(alternatives) < 2:
            skip_patterns = [
                "hey there", "i'll help", "i'll generate", "question:", "correct answer:",
                "the alternatives should", "please format", "api error", "connection error"]
            for line in response.strip().split('\n'):
                line_lower = line.strip().lower()
                # skip lines that are part of the instruction or too short
                if any(pattern in line_lower for pattern in skip_patterns) or len(line.strip()) < 10:
                    continue
                # skip lines that contain the correct answer
                if correct_answer.lower() in line_lower:
                    continue
                alternatives.append(line.strip())
                if len(alternatives) >= 2:
                    break
        # if still less than two alternatives, generate some generic options based on the question
        if len(alternatives) < 2:
            topic_words = question.split()
            topic_start = ' '.join(topic_words[:2]) if len(topic_words) >= 2 else question
            topic_end = ' '.join(topic_words[-2:]) if len(topic_words) >= 2 else question
            # list of generic but plausible incorrect options
            generic_options = [
                f"A common misconception about {topic_start}...",
                f"The opposite effect of {topic_end}...",
                f"A similar but unrelated concept to {topic_start}...",
                f"A methodology that conflicts with {topic_end}..."]
            for option in generic_options:
                # add generic option
                if option not in alternatives and option != correct_answer and len(alternatives) < 2:
                    alternatives.append(option)
        # return only the first two alternatives
        return alternatives[:2]
    except Exception as e:
        st.warning(f"Error generating alternatives with llama3: {str(e)}")
        return [
            "A related but incorrect concept",
            "A common misconception in this field"]


def save_quiz_to_outputs(questions, filename=None):
    try:
        records = []
        for q in questions:
            record = {
                'question': q['question'],
                'correct_answer': q['correct_answer'],
                'distractor_1': q['distractors'][0] if len(q['distractors']) > 0 else '',
                'distractor_2': q['distractors'][1] if len(q['distractors']) > 1 else '',
                'source': q.get('source', 'Unknown')}
            records.append(record)
        df = pd.DataFrame(records)
        output_dir = r"pages\quiz_outputs"
        os.makedirs(output_dir, exist_ok=True)
        if filename is None:
            book_title = st.session_state.get('selected_book')
            title_part = book_title
            existing_files = os.listdir(output_dir)
            quiz_numbers = []
            for f in existing_files:
                if f.startswith("quiz") and f.endswith(".csv") and "(" in f and ")" in f:
                    try:
                        num_part = f.split("(", 1)[0].replace("quiz", "")
                        if num_part.isdigit():
                            quiz_numbers.append(int(num_part))
                    except:
                        continue
            if quiz_numbers:
                next_number = max(quiz_numbers) + 1
            else:
                next_number = 1
            filename = f"quiz{next_number}({title_part}).csv"
        output_path = os.path.join(output_dir, filename)
        df.to_csv(output_path, index=False, encoding='utf-8-sig', sep=';')
        st.session_state.saved_csv_path = output_path
        return output_path
    except Exception as e:
        st.error(f"Failed to save quiz: {str(e)}")
        return None

def load_questions_from_csv():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(script_dir, '..', 'Contents', 'file.csv')
    try:
        df = pd.read_csv(csv_file_path, encoding='utf-8-sig',sep=';',on_bad_lines='skip',low_memory=False)
        book_column = None
        possible_book_columns = ['book title']
        for col in possible_book_columns:
            if col in df.columns:
                book_column = col
                break
        if book_column:
            books = df[book_column].unique().tolist()
            books = [book for book in books if book and str(book).strip() and str(book).lower() != 'nan']
            if books:
                books.insert(0, "All Books")
                if 'available_books' not in st.session_state:
                    st.session_state.available_books = books
                    st.session_state.book_column = book_column
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")
        return None


def generate_questions_from_selection(df, selected_book):
    questions = []
    if 'question' not in df.columns or 'answer' not in df.columns:
        st.error("CSV file must contain 'question' and 'answer' columns.")
        return questions
    book_column = st.session_state.get('book_column')
    if selected_book != "All Books" and book_column:
        df = df[df[book_column] == selected_book].reset_index(drop=True)
    total_questions = len(df)
    st.write(f"Generating questions from: {selected_book}")
    st.write(f"Number of available questions: {total_questions}")
    if total_questions == 0:
        st.warning("No questions available from the selected source. Please select a different source.")
        return questions
    progress_bar = st.progress(0)
    with st.spinner("Generating quiz alternatives with llama3..."):
        for i in range(len(df)):
            row = df.iloc[i]
            question_text = row['question']
            correct_answer = row['answer']
            distractors = generate_llm_alternatives(question_text, correct_answer)
            source = row.get(book_column, "Unknown") if book_column else row.get('source_pdf', "Unknown")
            question = {
                'question': question_text,
                'correct_answer': correct_answer,
                'distractors': distractors,
                'source': source }
            questions.append(question)
            progress_value = min(1.0, (i + 1) / total_questions)
            progress_bar.progress(progress_value)
    progress_bar.empty()
    return questions


def display_question(question, question_number, total_questions):
    st.markdown(f"## Question {question_number} of {total_questions}")
    st.markdown(f"### {question['question']}")
    options = [question['correct_answer']] + question['distractors']
    correct_option = options[0]
    random.shuffle(options)
    correct_index = options.index(correct_option)
    choice = st.radio(
        "Select your answer:",
        options,
        key=f"question_{question_number}")
    st.session_state.selected_option = options.index(choice)
    st.session_state.correct_option_index = correct_index
    while len(st.session_state.user_answers) < question_number:
        st.session_state.user_answers.append(None)
    st.session_state.user_answers[question_number - 1] = {
        'question': question['question'],
        'correct_answer': question['correct_answer'],
        'distractors': question['distractors'],
        'selected_answer': choice,
        'is_correct': (options.index(choice) == correct_index)}
    col1, col2 = st.columns(2)
    with col1:
        if question_number > 1:
            if st.button("Previous"):
                st.session_state.current_question_index -= 1
                st.rerun()
    with col2:
        if question_number < total_questions:
            submit_text = "Next"
        else:
            submit_text = "Submit Quiz"
        if st.button(submit_text):
            if st.session_state.selected_option == st.session_state.correct_option_index:
                st.session_state.score += 1
            if question_number < total_questions:
                st.session_state.current_question_index += 1
            else:
                st.session_state.quiz_in_progress = False
                st.session_state.quiz_completed = True
                save_quiz_to_outputs(st.session_state.quiz_questions)
            st.rerun()


def display_results():
    st.title("Quiz Results")
    total_questions = len(st.session_state.quiz_questions)
    score = st.session_state.score
    if total_questions > 0:
        percentage = (score / total_questions) * 100
    else:
        percentage = 0
    st.header(f"You scored: {score}/{total_questions} ({percentage:.1f}%)")
    if percentage >= 80:
        st.success("Great job! You really know this material well!")
        st.balloons()
    elif percentage >= 60:
        st.info("Good work! You have a solid understanding, but there's room for improvement.")
    else:
        st.warning("You might want to review the material again.")
    st.markdown("## Point Summary")
    col1, col2 = st.columns([3, 1])
    with col1:
        for i in range(total_questions):
            user_data = st.session_state.user_answers[i] if i < len(st.session_state.user_answers) else None
            is_correct = user_data.get('is_correct', False) if user_data else False
            st.markdown(f"**Question {i + 1}**: {'+1' if is_correct else '+0'} point")
    st.markdown("## Question Details")
    for i, q in enumerate(st.session_state.quiz_questions):
        user_data = st.session_state.user_answers[i] if i < len(st.session_state.user_answers) else None
        is_correct = user_data.get('is_correct', False) if user_data else False
        with st.expander(f"Question {i + 1}"):
            st.markdown(f"**Question:** {q['question']}")
            st.markdown(f"**Correct Answer:** {q['correct_answer']}")
            if user_data and 'selected_answer' in user_data:
                user_answer = user_data['selected_answer']
                st.markdown(f"**Your Answer:** {user_answer}")
            st.markdown("**Alternatives:**")
            for alt in q['distractors']:
                st.markdown(f"- {alt}")
            if 'source' in q and q['source'] != 'Unknown':
                st.markdown(f"**Source:** {q['source']}")
    if st.session_state.saved_csv_path:
        st.success(f"Quiz saved to: {st.session_state.saved_csv_path}")
    if st.button("Try Another Quiz", key="restart_btn", use_container_width=True):
        keys_to_clear = [
            'quiz_completed',
            'quiz_in_progress',
            'score',
            'user_answers',
            'quiz_questions',
            'current_question_index',
            'available_books',
            'book_column',
            'selected_book']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()


def simple_quiz_app():
    st.markdown("<h1 style='color:#96053e;'>Multiple-Choice Quiz</h1>", unsafe_allow_html=True)
    with st.sidebar:
        st.header("Quiz Settings")
        num_questions = st.slider("Number of Questions", min_value=3,max_value=10,value=5,step=1)
        st.session_state.num_questions = num_questions
    if st.session_state.get('quiz_completed', False):
        display_results()
    elif st.session_state.quiz_in_progress:
        current_idx = st.session_state.current_question_index
        total_questions = len(st.session_state.quiz_questions)
        if current_idx < total_questions:
            display_question(
                st.session_state.quiz_questions[current_idx],
                current_idx + 1,
                total_questions)
        else:
            st.session_state.quiz_in_progress = False
            st.session_state.quiz_completed = True
            st.rerun()
    else:
        st.markdown("This quiz generates questions with contextually relevant options using LLaMA3.")
        st.markdown(
            "After you complete the quiz, all questions, correct answers, and alternatives will be saved to a CSV file.")
        if 'available_books' not in st.session_state:
            df = load_questions_from_csv()
            if df is None:
                st.error("Failed to load question data. Please check the file path and format.")
                return
        if 'available_books' in st.session_state and st.session_state.available_books:
            selected_book = st.selectbox(
                "Select source material for questions:",
                st.session_state.available_books,
                index=0)
            st.session_state.selected_book = selected_book
            if st.button("Start Quiz", key="start_quiz_btn", use_container_width=True):
                with st.spinner("Generating your quiz questions..."):
                    df = load_questions_from_csv()
                    if df is not None:
                        all_questions = generate_questions_from_selection(df, selected_book)
                        if all_questions:
                            if len(all_questions) > num_questions:
                                st.session_state.quiz_questions = random.sample(all_questions, num_questions)
                            else:
                                st.session_state.quiz_questions = all_questions
                            st.session_state.current_question_index = 0
                            st.session_state.score = 0
                            st.session_state.user_answers = []
                            st.session_state.quiz_in_progress = True
                            st.session_state.quiz_completed = False
                            st.session_state.saved_csv_path = None
                            st.rerun()
        else:
            st.warning("No source books found in the CSV file. Please check the file format.")


if __name__ == "__main__":
    st.markdown(get_css(), unsafe_allow_html=True)
    simple_quiz_app()