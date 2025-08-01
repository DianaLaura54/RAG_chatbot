import os

import fitz
import subprocess
import csv
import re
import random
import unicodedata


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    text = unicodedata.normalize('NFC', text)
    return text


def chunk_text(text, max_words=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        if len(chunk.strip()) > 30:
            chunks.append(chunk.strip())
    return chunks


def make_prompt(text_chunk):
    clean_chunk = clean_text_for_prompt(text_chunk)
    return f"""Generate a question and answer from the following paragraph.
Paragraph:
{clean_chunk}
Q:"""

def clean_text_for_prompt(text):
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    return text


def call_llama3(prompt):
    try:
        encoded_prompt = prompt.encode('utf-8', errors='replace')
        result = subprocess.run(
            ['ollama', 'run', 'llama3'],
            input=encoded_prompt,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60
        )
        return result.stdout.decode('utf-8', errors='replace')
    except subprocess.TimeoutExpired:
        return "ERROR: Timeout"
    except Exception as e:
        return f"ERROR: {str(e)}"


def parse_qa(output):
    output = unicodedata.normalize('NFC', output)
    if "Paragraph:" in output:
        paragraph_pos = output.find("Paragraph:")
        q_pos = output.find("Q:", paragraph_pos)
        if q_pos != -1:
            output = output[q_pos:]
    lines = output.strip().splitlines()
    question = ""
    answer = ""
    current_section = None
    for line in lines:
        line_strip = line.strip()
        if line_strip.lower().startswith("q:"):
            current_section = "question"
            question = line_strip.split(":", 1)[1].strip() if ":" in line_strip else ""
        elif line_strip.lower().startswith("a:"):
            current_section = "answer"
            answer = line_strip.split(":", 1)[1].strip() if ":" in line_strip else ""
        elif current_section == "answer" and line_strip:
            answer += " " + line_strip
        elif current_section == "question" and line_strip:
            question += " " + line_strip
    question = re.sub(r'^["\']|["\']$', '', question).strip()
    answer = re.sub(r'^["\']|["\']$', '', answer).strip()
    question = re.sub(r'^Q[:.]\s*', '', question).strip()
    answer = re.sub(r'^A[:.]\s*', '', answer).strip()
    return question, answer


def generate_file_csv_format(pdf_path, output_path="file.csv"):
    book_title = os.path.splitext(os.path.basename(pdf_path))[0]
    print(f" Loading PDF: {pdf_path}")
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    successful_pairs = []
    target_pairs = 50
    if len(chunks) < target_pairs:
        print(f" Only {len(chunks)} chunks available. Adjusting chunk size to get more chunks...")
        new_max_words = max(50, int(150 * len(chunks) / target_pairs))
        chunks = chunk_text(text, max_words=new_max_words)
        print(f"After adjustment: {len(chunks)} chunks available.")
    random.shuffle(chunks)
    print(f" Generating {target_pairs} question-answer pairs...")
    chunk_index = 0
    attempts = 0
    max_attempts = target_pairs * 3
    while len(successful_pairs) < target_pairs and attempts < max_attempts:
        chunk = chunks[chunk_index % len(chunks)]
        chunk_index += 1
        attempts += 1
        print(f" Progress: {len(successful_pairs)}/{target_pairs} pairs generated (Attempt {attempts})...")
        prompt = make_prompt(chunk)
        response = call_llama3(prompt)
        if response.startswith("ERROR:"):
            print(f"    Skipping attempt - error calling model: {response}")
            continue
        question, answer = parse_qa(response)
        if not question or not answer:
            print(f"Skipping attempt - couldn't extract valid QA pair")
            continue
        duplicate = False
        for existing_tuple in successful_pairs:
            existing_q = existing_tuple[2]
            if existing_q.lower() == question.lower() or (
                    len(existing_q) > 10 and
                    (existing_q.lower() in question.lower() or question.lower() in existing_q.lower())):
                duplicate = True
                print(f"Skipping - detected similar question: {question[:40]}...")
                break
        if not duplicate:
            successful_pairs.append((book_title, chunk, question, answer))
            print(f"Added [{len(successful_pairs)}/50]: Q: {question[:50]}{'...' if len(question) > 50 else ''}")
    with open(output_path, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["context", "question", "answer", "book title"])
        for book_title, context, question, answer in successful_pairs[:target_pairs]:
            writer.writerow([context, question, answer, book_title])
    print(f"\n Final dataset saved in file.csv format at: {output_path}")
    print(f" Generated {len(successful_pairs)} QA pairs with context out of {target_pairs} target.")


if __name__ == "__main__":
    PDF_PATH = "Contents\\books\\A Christmas Carol.pdf"
    OUTPUT_CSV = "Contents\\books.csv"
    generate_file_csv_format(PDF_PATH, OUTPUT_CSV)