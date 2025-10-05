import string

import boto3
import json
import logging
import os
import time
from datetime import datetime
from typing import List, Dict, Optional, Any
from botocore.exceptions import ClientError
from nltk import word_tokenize
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class AWSTextractDocumentProcessor:
    def __init__(self, region_name: str = 'us-east-1'):
        self.textract_client = boto3.client('textract', region_name=region_name)
        self.s3_client = boto3.client('s3', region_name=region_name)

    def extract_text_from_document(self, bucket_name: str, document_key: str) -> List[tuple]:
        try:
            response = self.textract_client.start_document_text_detection(
                DocumentLocation={
                    'S3Object': {
                        'Bucket': bucket_name,
                        'Name': document_key
                    }
                }
            )
            job_id = response['JobId']
            max_wait_time = 300
            wait_time = 0
            while wait_time < max_wait_time:
                response = self.textract_client.get_document_text_detection(JobId=job_id)
                status = response['JobStatus']
                if status == 'SUCCEEDED':
                    break
                elif status == 'FAILED':
                    raise Exception(f"Textract job failed: {response.get('StatusMessage', '')}")
                time.sleep(2)
                wait_time += 2
            if wait_time >= max_wait_time:
                raise Exception("Textract job timed out")
            pages = {}
            blocks = response['Blocks']
            next_token = response.get('NextToken')
            while next_token:
                response = self.textract_client.get_document_text_detection(
                    JobId=job_id,
                    NextToken=next_token
                )
                blocks.extend(response['Blocks'])
                next_token = response.get('NextToken')
            for block in blocks:
                if block['BlockType'] == 'LINE':
                    page_num = block.get('Page', 1)
                    if page_num not in pages:
                        pages[page_num] = []
                    pages[page_num].append(block['Text'])
            result = []
            for page_num in sorted(pages.keys()):
                page_text = '\n'.join(pages[page_num])
                if page_text.strip():
                    result.append((page_text, page_num))
            logger.info(f"Extracted text from {len(result)} pages using Textract")
            return result
        except Exception as e:
            logger.error(f"Textract extraction failed for {document_key}: {str(e)}")
            raise


class AWSComprehendTextAnalyzer:
    def __init__(self, region_name: str = 'us-east-1'):
        self.comprehend_client = boto3.client('comprehend', region_name=region_name)

    def analyze_text_sentiment(self, text: str) -> Dict:
        try:
            if len(text.encode('utf-8')) > 5000:
                text = text[:4500]
            response = self.comprehend_client.detect_sentiment(
                Text=text,
                LanguageCode='en'
            )
            return {
                'sentiment': response['Sentiment'],
                'confidence_scores': response['SentimentScore']
            }
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {str(e)}")
            return {}

    def extract_key_phrases(self, text: str) -> List[str]:
        try:
            if len(text.encode('utf-8')) > 5000:
                text = text[:4500]
            response = self.comprehend_client.detect_key_phrases(
                Text=text,
                LanguageCode='en'
            )
            key_phrases = [phrase['Text'] for phrase in response['KeyPhrases']]
            return key_phrases
        except Exception as e:
            logger.error(f"Key phrase extraction failed: {str(e)}")
            return []

    def detect_entities(self, text: str) -> List[Dict]:
        try:
            if len(text.encode('utf-8')) > 5000:
                text = text[:4500]
            response = self.comprehend_client.detect_entities(
                Text=text,
                LanguageCode='en'
            )
            entities = []
            for entity in response['Entities']:
                entities.append({
                    'text': entity['Text'],
                    'type': entity['Type'],
                    'confidence': entity['Score']
                })
            return entities
        except Exception as e:
            logger.error(f"Entity detection failed: {str(e)}")
            return []


class AWSRekognitionImageProcessor:
    def __init__(self, region_name: str = 'us-east-1'):
        self.rekognition_client = boto3.client('rekognition', region_name=region_name)

    def extract_text_from_image(self, bucket_name: str, image_key: str) -> str:
        try:
            response = self.rekognition_client.detect_text(
                Image={
                    'S3Object': {
                        'Bucket': bucket_name,
                        'Name': image_key
                    }
                }
            )
            text_lines = []
            for text_detection in response['TextDetections']:
                if text_detection['Type'] == 'LINE':
                    text_lines.append(text_detection['DetectedText'])
            return '\n'.join(text_lines)
        except Exception as e:
            logger.error(f"Rekognition text detection failed for {image_key}: {str(e)}")
            return ""


class AWSCloudWatchLogger:
    def __init__(self, log_group_name: str, region_name: str = 'us-east-1'):
        self.cloudwatch_logs = boto3.client('logs', region_name=region_name)
        self.log_group_name = log_group_name
        self.log_stream_name = f"rag-system-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        try:
            self._ensure_log_group_exists()
            self._create_log_stream()
        except Exception as e:
            logger.warning(f"CloudWatch setup failed: {str(e)}")

    def _ensure_log_group_exists(self):
        try:
            self.cloudwatch_logs.describe_log_groups(logGroupNamePrefix=self.log_group_name)
        except ClientError:
            self.cloudwatch_logs.create_log_group(logGroupName=self.log_group_name)

    def _create_log_stream(self):
        try:
            self.cloudwatch_logs.create_log_stream(
                logGroupName=self.log_group_name,
                logStreamName=self.log_stream_name
            )
        except ClientError as e:
            if e.response['Error']['Code'] != 'ResourceAlreadyExistsException':
                raise

    def log_event(self, level: str, message: str, metadata: Dict = None):
        try:
            log_entry = {
                'timestamp': int(time.time() * 1000),
                'level': level,
                'message': message,
                'metadata': metadata or {}
            }
            self.cloudwatch_logs.put_log_events(
                logGroupName=self.log_group_name,
                logStreamName=self.log_stream_name,
                logEvents=[{
                    'timestamp': log_entry['timestamp'],
                    'message': json.dumps(log_entry)
                }]
            )
        except Exception as e:
            logger.error(f"CloudWatch logging failed: {str(e)}")


class ComprehensiveAWSRAGSystem:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        from aws_s3_integration import AWSS3DocumentManager
        self.s3_manager = AWSS3DocumentManager(
            config['s3_bucket'],
            config.get('region', 'us-east-1')
        )
        if config.get('use_textract'):
            self.textract_processor = AWSTextractDocumentProcessor(
                config.get('region', 'us-east-1')
            )
        if config.get('use_rekognition'):
            self.rekognition_processor = AWSRekognitionImageProcessor(
                config.get('region', 'us-east-1')
            )
        if config.get('use_comprehend'):
            self.comprehend_analyzer = AWSComprehendTextAnalyzer(
                config.get('region', 'us-east-1')
            )
        if config.get('use_cloudwatch') and config.get('cloudwatch_log_group'):
            self.cloudwatch_logger = AWSCloudWatchLogger(
                config['cloudwatch_log_group'],
                config.get('region', 'us-east-1')
            )

    def process_documents_with_aws(self, s3_prefix: str = "documents/") -> Dict[str, Any]:
        try:
            self._log_event('INFO', f'Starting document processing for prefix: {s3_prefix}')
            documents = self.s3_manager.list_documents(s3_prefix)
            all_documents_with_pages = []
            file_sources = []
            processing_errors = []
            for doc in documents:
                try:
                    self._log_event('INFO', f'Processing document: {doc["key"]}')
                    doc_pages = self._extract_text_with_pypdf2(doc)
                    if doc_pages:
                        enhanced_pages = []
                        for text, page_num in doc_pages:
                            enhanced_text = text
                            if self.config.get('use_comprehend'):
                                try:
                                    key_phrases = self.comprehend_analyzer.extract_key_phrases(text)
                                    entities = self.comprehend_analyzer.detect_entities(text)
                                    if key_phrases:
                                        enhanced_text += f"\n[Key Phrases: {', '.join(key_phrases[:5])}]"
                                    if entities:
                                        entity_names = [e['text'] for e in entities[:5]]
                                        enhanced_text += f"\n[Entities: {', '.join(entity_names)}]"
                                except Exception as comprehend_error:
                                    logger.warning(f"Comprehend analysis failed: {comprehend_error}")
                            enhanced_pages.append((enhanced_text, page_num))
                        all_documents_with_pages.append(enhanced_pages)
                        file_sources.append(doc['key'])
                    else:
                        processing_errors.append(f"No content extracted from {doc['filename']}")
                except Exception as e:
                    error_msg = f'Failed to process {doc["key"]}: {str(e)}'
                    self._log_event('ERROR', error_msg)
                    processing_errors.append(error_msg)
                    continue
            self._log_event('INFO', f'Successfully processed {len(all_documents_with_pages)} documents')
            result = {
                'documents_with_pages': all_documents_with_pages,
                'file_sources': file_sources,
                'total_documents': len(documents),
                'processed_documents': len(all_documents_with_pages),
                'processing_errors': processing_errors
            }
            self.documents_with_pages = all_documents_with_pages
            self.file_sources = file_sources
            return result
        except Exception as e:
            error_msg = f'Document processing failed: {str(e)}'
            self._log_event('ERROR', error_msg)
            raise

    def _extract_text_with_pypdf2(self, doc) -> List[tuple]:
        try:
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{doc['filename']}") as tmp_file:
                obj = self.s3_manager.s3_client.get_object(
                    Bucket=self.config['s3_bucket'],
                    Key=doc['key']
                )
                content = obj['Body'].read()
                tmp_file.write(content)
                tmp_file_path = tmp_file.name
            try:
                if doc['filename'].lower().endswith('.pdf'):
                    try:
                        import PyPDF2
                        with open(tmp_file_path, 'rb') as file:
                            pdf_reader = PyPDF2.PdfReader(file)
                            text_pages = []
                            for i, page in enumerate(pdf_reader.pages):
                                try:
                                    text = page.extract_text()
                                    if text.strip():
                                        text_pages.append((text, i + 1))
                                except Exception:
                                    continue
                            return text_pages
                    except ImportError:
                        logger.warning("PyPDF2 not available - install with 'pip install PyPDF2'")
                        return None
                    except Exception as pdf_error:
                        logger.warning(f"PyPDF2 extraction failed: {pdf_error}")
                        return None
                elif doc['filename'].lower().endswith(('.txt', '.md', '.csv')):
                    try:
                        with open(tmp_file_path, 'r', encoding='utf-8') as file:
                            content = file.read()
                            if content.strip():
                                return [(content, 1)]
                            else:
                                return None
                    except UnicodeDecodeError:
                        try:
                            with open(tmp_file_path, 'r', encoding='latin-1') as file:
                                content = file.read()
                                if content.strip():
                                    return [(content, 1)]
                        except Exception:
                            return None
                    except Exception:
                        return None
                elif doc['filename'].lower().endswith('.docx'):
                    try:
                        import docx
                        document = docx.Document(tmp_file_path)
                        text = []
                        for paragraph in document.paragraphs:
                            if paragraph.text.strip():
                                text.append(paragraph.text)
                        if text:
                            content = '\n'.join(text)
                            return [(content, 1)]
                        else:
                            return None
                    except ImportError:
                        logger.warning("python-docx not available - install with 'pip install python-docx'")
                        return None
                    except Exception:
                        return None
                else:
                    return None
            finally:
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
            return None
        except Exception:
            return None

    def enhanced_search_with_aws(self, query: str, documents_with_pages: List, **kwargs) -> Dict[str, Any]:
        try:
            enhanced_query = query
            query_analysis = {}
            if self.config.get('use_comprehend'):
                entities = self.comprehend_analyzer.detect_entities(query)
                key_phrases = self.comprehend_analyzer.extract_key_phrases(query)
                query_analysis = {
                    'entities': entities,
                    'key_phrases': key_phrases,
                    'original_query': query
                }
                if entities or key_phrases:
                    enhancement_terms = []
                    if entities:
                        enhancement_terms.extend([e['text'] for e in entities[:3]])
                    if key_phrases:
                        enhancement_terms.extend(key_phrases[:3])
                    enhanced_query = f"{query} {' '.join(enhancement_terms)}"
            search_results = self._perform_search(enhanced_query, documents_with_pages, **kwargs)
            return {
                'results': search_results,
                'query_analysis': query_analysis,
                'enhanced_query': enhanced_query
            }
        except Exception as e:
            self._log_event('ERROR', f'Enhanced search failed: {str(e)}')
            raise

    def _perform_search(self, query: str, documents_with_pages: List, **kwargs) -> List:
        try:
            from chunking import chunk_documents
            from embeddings import get_embedding_model, batch_generate_embeddings
            from faiss_index import create_faiss_index
            from lexical import create_bm25_index
            from search import hybrid_search, semantic_search
            file_sources = getattr(self, 'file_sources', [f'Document_{i}' for i in range(len(documents_with_pages))])
            chunks, metadata = chunk_documents(documents_with_pages, file_sources)
            if not chunks:
                return []
            embeddings = batch_generate_embeddings(chunks)
            index = create_faiss_index(embeddings, embeddings.shape[1])
            bm25_model, tokenized_corpus = create_bm25_index(chunks)
            results = hybrid_search(
                index=index,
                bm25_model=bm25_model,
                tokenized_corpus=tokenized_corpus,
                texts=chunks,
                metadata=metadata,
                query=query,
                n_semantic=7,
                n_lexical=5,
                alpha=0.7,
                n_results=5
            )
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'text': result['text'],
                    'content': result['text'],
                    'source': result['metadata'].get('source', 'Unknown'),
                    'filename': result['metadata'].get('source', 'Unknown').split('/')[-1],
                    'page': result['metadata'].get('page', 1),
                    'score': round(result['score'], 3),
                    'confidence': round(result['score'], 3),
                    'search_type': result.get('search_type', 'hybrid'),
                    'semantic_score': result.get('semantic_score', 0),
                    'lexical_score': result.get('lexical_score', 0)
                })
            return formatted_results
        except Exception as e:
            logger.warning(f"Semantic search failed: {str(e)}")
            return self._basic_fallback_search(query, documents_with_pages)

    def _basic_fallback_search(self, query: str, documents_with_pages: List) -> List:
        def _bm25_search(self, query: str, documents_with_pages: List) -> List:
            try:
                query_tokens = word_tokenize(query.lower())
                file_sources = getattr(self, 'file_sources', [])
                tokenized_documents = []
                for doc_pages in documents_with_pages:
                    tokenized_pages = []
                    for page_text, page_num in doc_pages:
                        page_text = page_text.lower()
                        page_text = page_text.translate(str.maketrans('', '', string.punctuation))
                        tokenized_pages.append(word_tokenize(page_text))
                    tokenized_documents.append(tokenized_pages)
                bm25 = []
                for doc_pages in tokenized_documents:
                    flattened_doc = [token for page in doc_pages for token in page]
                    bm25.append(flattened_doc)
                bm25_model = BM25Okapi(bm25)
                results = []
                for doc_idx, doc_pages in enumerate(documents_with_pages):
                    source_name = file_sources[doc_idx].split('/')[-1] if doc_idx < len(
                        file_sources) else f'Document {doc_idx + 1}'
                    for page_text, page_num in doc_pages:
                        page_tokens = word_tokenize(page_text.lower())
                        score = bm25_model.get_scores(query_tokens)[doc_idx]
                        if score > 0:
                            context = ' '.join(page_tokens[:50])
                            results.append({
                                'text': context,
                                'content': context,
                                'source': source_name,
                                'filename': source_name,
                                'page': page_num,
                                'score': round(score, 3),
                                'confidence': round(score, 3),
                                'search_type': 'bm25'
                            })
                results.sort(key=lambda x: x['score'], reverse=True)
                return results[:5]
            except Exception as e:
                logger.error(f"BM25 search failed: {str(e)}")
                return [{
                    'text': f"Search error: {str(e)}",
                    'content': f"Unable to search for '{query}' due to technical issues",
                    'source': 'system',
                    'score': 0,
                    'error': str(e)
                }]


    def _search_with_ollama_generation(self, query: str, documents_with_pages: List) -> str:
        try:
            search_results = self._perform_search(query, documents_with_pages)
            if not search_results:
                return f"No relevant information found for '{query}' in your documents."
            context_parts = []

            for i, result in enumerate(search_results[:3], 1):
                context_parts.append(f"Result {i} (from {result['filename']}):\n{result['text']}")
            context = "\n\n".join(context_parts)
            try:
                import requests
                prompt = f"""Based on the following context from documents, answer the question naturally and comprehensively. Context: {context}Question: {query} Answer based on the context above:"""
                response = requests.post(
                    'http://localhost:11434/api/generate',
                    json={
                        "model": "llama3",
                        "prompt": prompt,
                        "stream": False
                    },
                    headers={'Content-Type': 'application/json'},
                    timeout=30
                )
                if response.status_code == 200:
                    ollama_response = response.json().get('response', '')
                    if ollama_response.strip():
                        return f"**AI-Generated Answer:**\n\n{ollama_response}\n\n**Sources:** {', '.join([r['filename'] for r in search_results[:3]])}"
            except Exception as ollama_error:
                logger.warning(f"Ollama generation failed: {ollama_error}")
            response_parts = [f"**Search Results for: '{query}'**\n"]

            for i, result in enumerate(search_results[:3], 1):
                response_parts.append(f"**{i}. {result['filename']}** (Score: {result['score']})")
                response_parts.append(f"{result['text']}\n")
            return "\n".join(response_parts)
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def _log_event(self, level: str, message: str, metadata: Dict = None):
        if hasattr(self, 'cloudwatch_logger'):
            self.cloudwatch_logger.log_event(level, message, metadata)
        else:
            getattr(logger, level.lower())(message)


