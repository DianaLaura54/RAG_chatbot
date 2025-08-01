
import os
import pandas as pd
from typing import Dict, List, Any, Tuple
import csv
from chunking import chunk_documents, chunk_documents_semantic
from embeddings import AVAILABLE_EMBEDDING_MODELS, set_embedding_model, batch_generate_embeddings
from generation import query_llama3, query_mistral
from processing import get_all_files_in_folder, process_files, get_faiss_file_paths, get_bm25_file_paths
from search import hybrid_search, bm25_search, semantic_search
from faiss_index import (
    load_faiss_data,
    create_faiss_index,
    save_faiss_data,
)
from common import (
    log_max_bertscore_to_csv,
    format_source_info,
    optimize_query
)
from evaluate import (
    compute_bert_score,
    compute_bertscore_with_filter,
    filter_chunks_by_pdf,
    compute_rougel_with_filter,
    compute_rouge_l_score,
)
from lexical import (
    create_bm25_index,
    save_data,
    load_data
)
from reranking import reranker, get_available_reranker_models, get_default_reranker_model


class SingleQuestionTester:
    def __init__(self, base_path: str = r"Contents"):
        self.base_path = base_path
        self.folder_path = os.path.join(base_path, "books")
        self.csv_path = os.path.join(base_path, "file.csv")
        self.test_question = "Who offered to help retrieve the golden ball?"
        self.test_answer = "A Frog who stretched his thick ugly head out of the water."
        self.book_title = "The Frog Prince"
        print(f"Testing question: '{self.test_question}'")
        print(f"Expected answer: '{self.test_answer}'")
        print(f"Book title: '{self.book_title}' (The Frog Prince)")
        print(f"Will test against:")
        print(f"  - 'All PDFs' (search across all books)")
        print(f"  - 'The Frog Prince")
        self.embedding_models = list(AVAILABLE_EMBEDDING_MODELS.keys())[:2]
        self.chunking_methods = ["standard", "semantic"]
        self.search_methods = ["semantic", "lexical", "hybrid"]
        self.llm_models = ["llama3", "mistral"]
        self.reranker_options = [True, False]
        self.query_optimization_options = [True, False]
        self.reranker_models = list(get_available_reranker_models())
        self.alpha_values = [0.5]
        self.n_semantic_values = [5]
        self.n_lexical_values = [3]
        self.available_pdfs = self._get_available_pdfs()

    def _get_available_pdfs(self) -> List[str]:
        try:
            pdf_files = [f for f in os.listdir(self.folder_path) if f.lower().endswith('.pdf')]
            def extract_number(filename):
                try:
                    return int(filename.split('.')[0])
                except:
                    return float('inf')
            pdf_files.sort(key=extract_number)
            pdf_options = ["All PDFs"]
            target_pdf = "The Frog Prince.pdf"
            if target_pdf in pdf_files:
                pdf_options.append(target_pdf)
            else:
                print(f"Warning: {target_pdf} not found in manuals folder")
            print(f"Found PDF files: {pdf_files}")
            print(f"Will test with: {pdf_options}")
            return pdf_options
        except Exception as e:
            print(f"Error getting PDF files: {str(e)}")
            return ["All PDFs"]

    def _get_expected_pdf_for_question(self) -> str:
        return "The Frog Prince.pdf"

    def _ensure_indices_exist(self, chunking_method: str, embedding_model: str) -> bool:
        faiss_path, _, _, _ = get_faiss_file_paths(chunking_method, embedding_model)
        bm25_path, _, _, _ = get_bm25_file_paths(chunking_method, embedding_model)
        if os.path.exists(faiss_path) and os.path.exists(bm25_path):
            return True
        print(f"Creating indices for {chunking_method} chunking with {embedding_model} embeddings...")
        try:
            set_embedding_model(embedding_model)
            all_files = get_all_files_in_folder(self.folder_path)
            if not all_files:
                print("No files found in folder")
                return False
            all_documents_with_pages, file_sources = [], []
            for file_path in all_files:
                file_docs = process_files(file_path)
                if file_docs:
                    all_documents_with_pages.append(file_docs)
                    file_sources.append(file_path)
            if not all_documents_with_pages:
                print("No documents could be processed")
                return False
            if chunking_method == "semantic":
                try:
                    chunks, metadata = chunk_documents_semantic(all_documents_with_pages, file_sources)
                except Exception as e:
                    print(f"Semantic chunking failed, using standard: {str(e)}")
                    chunks, metadata = chunk_documents(all_documents_with_pages, file_sources)
            else:
                chunks, metadata = chunk_documents(all_documents_with_pages, file_sources)
            embeddings = batch_generate_embeddings(chunks, model_name=embedding_model)
            index = create_faiss_index(embeddings, embeddings.shape[1])
            save_faiss_data(index, embeddings, chunks, metadata, chunking_method=chunking_method, model_name=embedding_model)
            bm25_model, tokenized_corpus = create_bm25_index(chunks)
            save_data(bm25_model, tokenized_corpus, chunks, metadata,chunking_method, embedding_model)
            print(f"Successfully created indices for {chunking_method}/{embedding_model}")
            return True
        except Exception as e:
            print(f"Error creating indices: {str(e)}")
            return False


    def _load_search_data(self, chunking_method: str, embedding_model: str) -> Dict[str, Any]:
        try:
            set_embedding_model(embedding_model)
            index, embeddings, texts, metadata = load_faiss_data(chunking_method, embedding_model)
            bm25_model, tokenized_corpus, bm25_texts, bm25_metadata = load_data(chunking_method, embedding_model)
            if index is not None and texts and bm25_model and tokenized_corpus:
                final_texts = texts if bm25_texts is None else bm25_texts
                final_metadata = metadata if bm25_metadata is None else bm25_metadata
                return {
                    'faiss_index': index,
                    'texts': final_texts,
                    'metadata': final_metadata,
                    'bm25_model': bm25_model,
                    'tokenized_corpus': tokenized_corpus,
                    'current_chunking_method': chunking_method,
                    'embeddings': embeddings,
                    'embedding_model_name': embedding_model
                }
            else:
                print(f"Failed to load search data for {chunking_method}/{embedding_model}")
                return None
        except Exception as e:
            print(f"Error loading search data: {str(e)}")
            return None


    def _perform_search(self, query: str, search_data: Dict[str, Any], config: Dict[str, Any],selected_pdf: str = "All PDFs") -> List[Dict]:
        search_method = config['search_method']
        num_results = 5
        initial_num_results = num_results * 3 if config['use_reranker'] else num_results
        if search_method == "semantic":
            results = semantic_search(
                search_data['faiss_index'],
                search_data['texts'],
                search_data['metadata'],
                query,
                n_results=initial_num_results,
                model_name=search_data['embedding_model_name']
            )
        elif search_method == "lexical":
            results = bm25_search(
                search_data['bm25_model'],
                search_data['tokenized_corpus'],
                search_data['texts'],
                search_data['metadata'],
                query,
                n_results=initial_num_results
            )
        else:
            chunking_method = search_data.get('current_chunking_method', 'standard')
            alpha = config['alpha']
            if chunking_method == "semantic":
                alpha = min(alpha + 0.1, 0.9)
            results = hybrid_search(
                search_data['faiss_index'],
                search_data['bm25_model'],
                search_data['tokenized_corpus'],
                search_data['texts'],
                search_data['metadata'],
                query,
                n_semantic=config['n_semantic'],
                n_lexical=config['n_lexical'],
                alpha=alpha,
                n_results=initial_num_results,
                model_name=search_data['embedding_model_name']
            )
        if selected_pdf != "All PDFs" and results:
            filtered_results = filter_chunks_by_pdf(results, selected_pdf)
            if filtered_results:
                results = filtered_results
        if config['use_reranker'] and results:
            results = reranker(query, results, config['reranker_model'])[:num_results]
        else:
            results = results[:num_results]
        return results

    def _generate_response(self, query: str, chunks: List[Dict], llm_model: str) -> str:
        if not chunks:
            return "No relevant documents found."
        prompt_template = (
            "Hey there! I'll help you find the answer to your question based on these stories:\n"
            "{relevant_documents}\n\n"
            "Here's your question: {user_input}\n"
            "If the answer isn't in the stories, I'll just say 'I don't know'."
        )
        if llm_model == "mistral":
            return query_mistral(prompt_template, query, chunks)
        else:
            return query_llama3(prompt_template, query, chunks)

    def _evaluate_response(self, chunks: List[Dict], response: str, actual_answer: str) -> Dict[str, float]:
        if not chunks:
            return {
                'bert_score': 0.0,
                'rouge_l_score': 0.0,
                'response_answer_bert_score': 0.0,
                'max_chunk_answer_bert_score': 0.0,
                'response_answer_rouge_score': 0.0,
                'max_chunk_answer_rouge_score': 0.0
            }
        bert_chunk_scores, chunks_used = compute_bertscore_with_filter(chunks, response)
        bert_score = max(bert_chunk_scores) if bert_chunk_scores else 0.0
        rouge_chunk_scores, _ = compute_rougel_with_filter(chunks, response)
        rouge_l_score = max(rouge_chunk_scores) if rouge_chunk_scores else 0.0
        answer_chunk = [{"text": actual_answer}]
        response_answer_bert_score = compute_bert_score(answer_chunk, response)
        response_answer_rouge_score = compute_rouge_l_score(answer_chunk, response)
        bert_chunk_answer_scores = []
        rouge_chunk_answer_scores = []
        for chunk in chunks:
            bert_chunk_answer_scores.append(compute_bert_score(answer_chunk, chunk["text"]))
            rouge_chunk_answer_scores.append(compute_rouge_l_score(answer_chunk, chunk["text"]))
        max_chunk_answer_bert_score = max(bert_chunk_answer_scores) if bert_chunk_answer_scores else 0.0
        max_chunk_answer_rouge_score = max(rouge_chunk_answer_scores) if rouge_chunk_answer_scores else 0.0
        return {
            'bert_score': bert_score,
            'rouge_l_score': rouge_l_score,
            'response_answer_bert_score': response_answer_bert_score,
            'max_chunk_answer_bert_score': max_chunk_answer_bert_score,
            'response_answer_rouge_score': response_answer_rouge_score,
            'max_chunk_answer_rouge_score': max_chunk_answer_rouge_score
        }

    def _log_results(self, question: str, response: str, actual_answer: str, metrics: Dict[str, float], config: Dict[str, Any], selected_pdf: str):
        csv_file = os.path.join(self.base_path, "scores_log.csv")
        def clean_text(text):
            if isinstance(text, str):
                text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
                text = ' '.join(text.split())
                text = text.replace('"', '""')
            return text
        row_data = {
            'question': clean_text(question),
            'response': clean_text(response),
            'answer': clean_text(actual_answer),
            'selected_pdf': selected_pdf,
            'LLM Model': config['llm_model'],
            'Search Type': config['search_method'],
            'ResponseChunkBERTScore': round(metrics['bert_score'], 6),
            'ResponseChunkRougeL': round(metrics['rouge_l_score'], 6),
            'ResponseAnswerBERTScore': round(metrics['response_answer_bert_score'], 6),
            'ResponseAnswerRougeL': round(metrics['response_answer_rouge_score'], 6),
            'ChunkAnswerBERTScore': round(metrics['max_chunk_answer_bert_score'], 6),
            'ChunkAnswerRougeL': round(metrics['max_chunk_answer_rouge_score'], 6),
            'Reranker Used': str(config['use_reranker']),
            'Reranker Model': str(config.get('reranker_model', 'none')),
            'Chunking Method': config['chunking_method'],
            'QueryOptimization': str(config['use_query_optimization']),
            'Embedding Model': config['embedding_model']
        }
        file_exists = os.path.exists(csv_file)
        with open(csv_file, 'a', newline='', encoding='utf-8-sig') as f:
            fieldnames = [
                'question', 'response', 'answer', 'selected_pdf', 'LLM Model', 'Search Type',
                'ResponseChunkBERTScore', 'ResponseChunkRougeL', 'ResponseAnswerBERTScore',
                'ResponseAnswerRougeL', 'ChunkAnswerBERTScore', 'ChunkAnswerRougeL',
                'Reranker Used', 'Reranker Model', 'Chunking Method', 'QueryOptimization',
                'Embedding Model'
            ]
            writer = csv.DictWriter(f,fieldnames=fieldnames,delimiter=';',quotechar='"',quoting=csv.QUOTE_MINIMAL,lineterminator='\n')
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_data)
        print(f"    Logged to CSV: {csv_file}")


    def _log_results_pandas(self, question: str, response: str, actual_answer: str,metrics: Dict[str, float], config: Dict[str, Any], selected_pdf: str):
        csv_file = os.path.join(self.base_path, "scores_log.csv")
        new_row = {
            'question': question,
            'response': response,
            'answer': actual_answer,
            'selected_pdf': selected_pdf,
            'LLM Model': config['llm_model'],
            'Search Type': config['search_method'],
            'ResponseChunkBERTScore': round(metrics['bert_score'], 6),
            'ResponseChunkRougeL': round(metrics['rouge_l_score'], 6),
            'ResponseAnswerBERTScore': round(metrics['response_answer_bert_score'], 6),
            'ResponseAnswerRougeL': round(metrics['response_answer_rouge_score'], 6),
            'ChunkAnswerBERTScore': round(metrics['max_chunk_answer_bert_score'], 6),
            'ChunkAnswerRougeL': round(metrics['max_chunk_answer_rouge_score'], 6),
            'Reranker Used': config['use_reranker'],
            'Reranker Model': config.get('reranker_model', 'none'),
            'Chunking Method': config['chunking_method'],
            'QueryOptimization': config['use_query_optimization'],
            'Embedding Model': config['embedding_model']
        }
        new_df = pd.DataFrame([new_row])
        if os.path.exists(csv_file):
            new_df.to_csv(csv_file, mode='a', header=False, index=False, encoding='utf-8-sig')
        else:
            new_df.to_csv(csv_file, mode='w', header=True, index=False, encoding='utf-8-sig')
        print(f"    Logged to CSV: {csv_file}")



    def _generate_configurations(self) -> List[Dict[str, Any]]:
        configs = []
        for embedding_model in self.embedding_models:
            for chunking_method in self.chunking_methods:
                for search_method in self.search_methods:
                    for llm_model in self.llm_models:
                        for use_reranker in self.reranker_options:
                            for use_query_opt in self.query_optimization_options:
                                base_config = {
                                    'embedding_model': embedding_model,
                                    'chunking_method': chunking_method,
                                    'search_method': search_method,
                                    'llm_model': llm_model,
                                    'use_reranker': use_reranker,
                                    'use_query_optimization': use_query_opt,
                                    'reranker_model': get_default_reranker_model() if use_reranker else None
                                }
                                if search_method == "hybrid":
                                    for alpha in self.alpha_values:
                                        for n_semantic in self.n_semantic_values:
                                            for n_lexical in self.n_lexical_values:
                                                config = base_config.copy()
                                                config.update({
                                                    'alpha': alpha,
                                                    'n_semantic': n_semantic,
                                                    'n_lexical': n_lexical
                                                })
                                                configs.append(config)
                                else:
                                    config = base_config.copy()
                                    config.update({
                                        'alpha': 0.7,
                                        'n_semantic': 7,
                                        'n_lexical': 5
                                    })
                                    configs.append(config)
        return configs

    def run_test(self):
        print(f"\n{'=' * 80}")
        print("SINGLE QUESTION RAG TEST")
        print(f"Question from Book {self.book_title}: {self.test_question}")
        print(f"Expected Answer: {self.test_answer}")
        print(f"Expected PDF: {self._get_expected_pdf_for_question()}")
        print(f"{'=' * 80}")
        configs = self._generate_configurations()
        total_configs = len(configs)
        print(f"\nGenerated {total_configs} configuration combinations")
        print(f"PDF testing strategy:")
        print(f"  - All PDFs: Search across all books")
        print(f"  - The Frog Prince")
        print(f"  - Expected best result: The Frog Prince.pdf (since question is from book The Frog Prince)")
        total_tests = total_configs * len(self.available_pdfs)
        print(f"Total tests: {total_tests}")
        current_test = 0
        results_summary = []
        for config_idx, config in enumerate(configs):
            print(f"\n{'-' * 60}")
            print(f"Configuration {config_idx + 1}/{total_configs}")
            print(f"Embedding: {config['embedding_model']}")
            print(f"Chunking: {config['chunking_method']}")
            print(f"Search: {config['search_method']}")
            print(f"LLM: {config['llm_model']}")
            print(f"Reranker: {config['use_reranker']}")
            print(f"Query Opt: {config['use_query_optimization']}")
            if config['search_method'] == 'hybrid':
                print(f"Alpha: {config['alpha']}, Semantic: {config['n_semantic']}, Lexical: {config['n_lexical']}")
            if not self._ensure_indices_exist(config['chunking_method'], config['embedding_model']):
                print(f"Skipping configuration due to index creation failure")
                current_test += len(self.available_pdfs)
                continue
            search_data = self._load_search_data(config['chunking_method'], config['embedding_model'])
            if search_data is None:
                print(f"Skipping configuration due to data loading failure")
                current_test += len(self.available_pdfs)
                continue
            for pdf_option in self.available_pdfs:
                current_test += 1
                pdf_note = ""
                if pdf_option == self._get_expected_pdf_for_question():
                    pdf_note = " (EXPECTED BEST)"
                elif pdf_option == "All PDFs":
                    pdf_note = " (ALL BOOKS)"
                print(f"\n  Test {current_test}/{total_tests} - PDF: {pdf_option}{pdf_note}")
                try:
                    search_query = self.test_question
                    if config['use_query_optimization']:
                        optimized_queries, _ = optimize_query(self.test_question)
                        if optimized_queries and len(optimized_queries) > 1:
                            search_query = optimized_queries[1]
                            print(f"    Optimized query: {search_query}")
                    chunks = self._perform_search(search_query, search_data, config, pdf_option)
                    print(f"    Found {len(chunks)} chunks")
                    response = self._generate_response(search_query, chunks, config['llm_model'])
                    print(f"    Response: {response[:100]}...")
                    metrics = self._evaluate_response(chunks, response, self.test_answer)
                    self._log_results(self.test_question, response, self.test_answer, metrics, config, pdf_option)
                    result = {
                        'config_idx': config_idx + 1,
                        'pdf': pdf_option,
                        'embedding': config['embedding_model'],
                        'chunking': config['chunking_method'],
                        'search': config['search_method'],
                        'llm': config['llm_model'],
                        'reranker': config['use_reranker'],
                        'query_opt': config['use_query_optimization'],
                        'bert_score': metrics['bert_score'],
                        'rouge_l_score': metrics['rouge_l_score'],
                        'response_answer_bert': metrics['response_answer_bert_score'],
                        'response': response,
                        'is_expected_pdf': pdf_option == self._get_expected_pdf_for_question(),
                        'is_all_pdfs': pdf_option == "All PDFs"
                    }
                    if config['search_method'] == 'hybrid':
                        result['alpha'] = config['alpha']
                    results_summary.append(result)
                    print(f"    BERTScore: {metrics['bert_score']:.4f}")
                    print(f"    Rouge-L: {metrics['rouge_l_score']:.4f}")
                    print(f"    Answer BERTScore: {metrics['response_answer_bert_score']:.4f}")
                except Exception as e:
                    print(f"    Error: {str(e)}")
                    continue


def main():
    tester = SingleQuestionTester()
    tester.run_test()

if __name__ == "__main__":
    main()