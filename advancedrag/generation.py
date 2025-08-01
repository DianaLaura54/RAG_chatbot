import requests


def query_llama3(prompt, user_input, relevant_chunks):
    url = 'http://localhost:11434/api/generate'
    relevant_texts = [chunk["text"] for chunk in relevant_chunks if isinstance(chunk, dict) and "text" in chunk] if all(isinstance(chunk, dict) and "text" in chunk for chunk in relevant_chunks) else relevant_chunks
    data = {
        "model": "llama3:8b",
        "prompt": prompt.format(user_input=user_input, relevant_documents="\n".join(relevant_texts)),
        "stream": False
    }
    try:
        response = requests.post(url, json=data, headers={'Content-Type': 'application/json'})
        return response.json().get('response', 'No response found') if response.status_code == 200 else f"API Error: {response.status_code}"
    except Exception as e:
        return "API connection error"

def query_mistral(prompt, user_input, relevant_chunks):
    url = 'http://localhost:11434/api/generate'
    relevant_texts = [chunk["text"] for chunk in relevant_chunks if isinstance(chunk, dict) and "text" in chunk] if all(isinstance(chunk, dict) and "text" in chunk for chunk in relevant_chunks) else relevant_chunks
    data = {
        "model": "mistral",
        "prompt": prompt.format(user_input=user_input, relevant_documents="\n".join(relevant_texts)),
        "stream": False
    }
    try:
        response = requests.post(url, json=data, headers={'Content-Type': 'application/json'})
        return response.json().get('response', 'No response found') if response.status_code == 200 else f"API Error: {response.status_code}"
    except Exception as e:
        return "API connection error"