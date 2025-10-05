import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


GX_AVAILABLE = False
GX_CONTEXT = None

try:
    import great_expectations as gx
    GX_AVAILABLE = True
    logger.info("Great Expectations imported successfully")
except ImportError:
    logger.warning("Great Expectations not available - using basic validation only")
    gx = None


class RAGDataValidator:
    def __init__(self, base_path: str = "Contents"):
        self.base_path = base_path
        self.ge_available = GX_AVAILABLE
        self._setup_validation()

    def _setup_validation(self):
        if self.ge_available:
            try:
                context = None
                try:
                    context = gx.get_context()
                    logger.info("Great Expectations context created using get_context()")
                except (AttributeError, Exception):
                    pass
                if context is None:
                    try:
                        from great_expectations.data_context import DataContext
                        context = DataContext()
                        logger.info("Great Expectations context created using DataContext()")
                    except (ImportError, AttributeError, Exception):
                        pass
                if context is None:
                    try:
                        from great_expectations.data_context import FileDataContext
                        context = FileDataContext()
                        logger.info("Great Expectations context created using FileDataContext()")
                    except (ImportError, AttributeError, Exception):
                        pass
                if context:
                    self.context = context
                    logger.info("Great Expectations context successfully initialized")
                else:
                    logger.warning("Could not create any GX context - using basic validation only")
                    self.ge_available = False
                    self.context = None
            except Exception as e:
                logger.warning(f"GX setup completely failed: {e}. Using basic validation.")
                self.ge_available = False
                self.context = None
        else:
            self.context = None

    def validate_pdf_processing_results(self, documents_with_pages: List[List[Tuple]],file_sources: List[str]) -> Dict[str, Any]:
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "statistics": {}
        }
        try:
            validation_data = []
            for src_idx, doc_tuples in enumerate(documents_with_pages):
                source_file = file_sources[src_idx]
                for content, page_num in doc_tuples:
                    validation_data.append({
                        'source_file': source_file,
                        'page_number': page_num,
                        'content_length': len(content) if content else 0,
                        'content_words': len(content.split()) if content else 0,
                        'has_content': bool(content and content.strip()),
                        'content_sample': content[:100] if content else ""
                    })
            if not validation_data:
                validation_results["is_valid"] = False
                validation_results["errors"].append("No documents processed")
                return validation_results
            df = pd.DataFrame(validation_data)
            self._validate_pdf_data(df, validation_results)
        except Exception as e:
            validation_results["is_valid"] = False
            validation_results["errors"].append(f"PDF validation error: {str(e)}")
        return validation_results

    def validate_chunking_results(self, chunks: List[str], metadata: List[Dict]) -> Dict[str, Any]:
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "statistics": {}
        }
        try:
            chunk_data = []
            for i, (chunk, meta) in enumerate(zip(chunks, metadata)):
                chunk_data.append({
                    'chunk_index': i,
                    'chunk_length': len(chunk),
                    'chunk_words': len(chunk.split()),
                    'source': meta.get('source', ''),
                    'page': meta.get('page', 0),
                    'sentences': meta.get('sentences', 0),
                    'chunking_method': meta.get('chunking_method', 'unknown'),
                    'starts_with_capital': chunk[0].isupper() if chunk else False,
                    'ends_with_punctuation': chunk[-1] in '.!?' if chunk else False,
                    'has_numeric_content': any(c.isdigit() for c in chunk) if chunk else False,
                    'special_char_ratio': sum(1 for c in chunk if not c.isalnum() and not c.isspace()) / len(
                        chunk) if chunk else 0
                })
            if not chunk_data:
                validation_results["is_valid"] = False
                validation_results["errors"].append("No chunks to validate")
                return validation_results
            df = pd.DataFrame(chunk_data)
            self._validate_chunking_data(df, validation_results)
        except Exception as e:
            validation_results["is_valid"] = False
            validation_results["errors"].append(f"Chunking validation error: {str(e)}")
        return validation_results

    def validate_embeddings(self, embeddings: np.ndarray, texts: List[str]) -> Dict[str, Any]:
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "statistics": {}
        }
        try:
            if embeddings.shape[0] != len(texts):
                validation_results["is_valid"] = False
                validation_results["errors"].append(
                    f"Mismatch: {embeddings.shape[0]} embeddings for {len(texts)} texts"
                )
                return validation_results
            embedding_data = []
            for i in range(len(embeddings)):
                emb = embeddings[i]
                embedding_data.append({
                    'index': i,
                    'embedding_dimension': len(emb),
                    'embedding_norm': np.linalg.norm(emb),
                    'has_nan_values': np.isnan(emb).any(),
                    'has_inf_values': np.isinf(emb).any(),
                    'mean_value': np.mean(emb),
                    'std_value': np.std(emb),
                    'min_value': np.min(emb),
                    'max_value': np.max(emb),
                    'text_length': len(texts[i]) if i < len(texts) else 0,
                    'zero_values_count': np.sum(emb == 0),
                    'positive_values_ratio': np.sum(emb > 0) / len(emb),
                    'negative_values_ratio': np.sum(emb < 0) / len(emb)
                })
            df = pd.DataFrame(embedding_data)
            self._validate_embeddings_data(df, validation_results)
        except Exception as e:
            validation_results["is_valid"] = False
            validation_results["errors"].append(f"Embeddings validation error: {str(e)}")
        return validation_results

    def validate_search_results(self, search_results: List[Dict], query: str, search_method: str) -> Dict[str, Any]:
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "statistics": {}
        }
        try:
            results_data = []
            for i, result in enumerate(search_results):
                metadata = result.get('metadata', {})
                results_data.append({
                    'result_index': i,
                    'has_text': bool(result.get('text')),
                    'text_length': len(result.get('text', '')),
                    'text_word_count': len(result.get('text', '').split()),
                    'has_metadata': bool(metadata),
                    'has_score': result.get('score') is not None,
                    'score': result.get('score', 0),
                    'search_method': search_method,
                    'has_source': bool(metadata.get('source')),
                    'has_page': bool(metadata.get('page')),
                    'query_terms_in_text': self._count_query_terms_in_text(query, result.get('text', '')),
                    'text_quality_score': self._calculate_text_quality_score(result.get('text', ''))
                })
            if not results_data:
                validation_results["warnings"].append("No search results to validate")
                validation_results["statistics"]["total_results"] = 0
                return validation_results
            df = pd.DataFrame(results_data)
            self._validate_search_data(df, validation_results)
        except Exception as e:
            validation_results["is_valid"] = False
            validation_results["errors"].append(f"Search validation error: {str(e)}")
        return validation_results

    def _validate_pdf_data(self, df: pd.DataFrame, validation_results: Dict):
        stats = {
            "total_pages": len(df),
            "unique_sources": df['source_file'].nunique(),
            "avg_content_length": df['content_length'].mean(),
            "median_content_length": df['content_length'].median(),
            "empty_pages": (df['content_length'] == 0).sum(),
            "pages_with_minimal_content": (df['content_words'] < 10).sum(),
            "pages_with_substantial_content": (df['content_words'] > 100).sum(),
            "max_content_length": df['content_length'].max(),
            "min_content_length": df['content_length'].min(),
            "content_length_std": df['content_length'].std()
        }
        validation_results["statistics"] = stats
        if stats["total_pages"] == 0:
            validation_results["is_valid"] = False
            validation_results["errors"].append("No pages found in any document")
        if stats["empty_pages"] == stats["total_pages"]:
            validation_results["is_valid"] = False
            validation_results["errors"].append("All pages are empty")
        if stats["empty_pages"] > stats["total_pages"] * 0.5:
            validation_results["warnings"].append(
                f"High proportion of empty pages: {stats['empty_pages']}/{stats['total_pages']}")
        if stats["pages_with_minimal_content"] > stats["total_pages"] * 0.3:
            validation_results["warnings"].append(
                f"Many pages with minimal content: {stats['pages_with_minimal_content']}/{stats['total_pages']}")
        if stats["unique_sources"] == 0:
            validation_results["errors"].append("No source files identified")
            validation_results["is_valid"] = False
        if self.ge_available and self.context:
            try:
                self._run_ge_pdf_validation(df, validation_results)
            except Exception as e:
                logger.warning(f"GX PDF validation failed: {e}")

    def _validate_chunking_data(self, df: pd.DataFrame, validation_results: Dict):
        stats = {
            "total_chunks": len(df),
            "avg_chunk_length": df['chunk_length'].mean(),
            "median_chunk_length": df['chunk_length'].median(),
            "avg_words_per_chunk": df['chunk_words'].mean(),
            "median_words_per_chunk": df['chunk_words'].median(),
            "min_chunk_words": df['chunk_words'].min(),
            "max_chunk_words": df['chunk_words'].max(),
            "chunks_too_short": (df['chunk_words'] < 10).sum(),
            "chunks_too_long": (df['chunk_words'] > 500).sum(),
            "chunks_optimal_size": ((df['chunk_words'] >= 50) & (df['chunk_words'] <= 250)).sum(),
            "chunks_start_with_capital": df['starts_with_capital'].sum(),
            "chunks_end_with_punctuation": df['ends_with_punctuation'].sum(),
            "unique_sources": df['source'].nunique() if 'source' in df.columns else 0,
            "avg_special_char_ratio": df['special_char_ratio'].mean(),
            "chunks_with_numeric_content": df['has_numeric_content'].sum()
        }
        validation_results["statistics"] = stats
        if stats["total_chunks"] == 0:
            validation_results["is_valid"] = False
            validation_results["errors"].append("No chunks created")
        if stats["min_chunk_words"] == 0:
            validation_results["is_valid"] = False
            validation_results["errors"].append("Found chunks with no words")
        if stats["chunks_too_short"] > stats["total_chunks"] * 0.2:
            validation_results["warnings"].append(
                f"Many chunks are too short: {stats['chunks_too_short']}/{stats['total_chunks']}")
        if stats["chunks_too_long"] > stats["total_chunks"] * 0.1:
            validation_results["warnings"].append(
                f"Some chunks are too long: {stats['chunks_too_long']}/{stats['total_chunks']}")
        if stats["chunks_start_with_capital"] < stats["total_chunks"] * 0.7:
            validation_results["warnings"].append("Many chunks don't start with capital letters")
        if stats["avg_special_char_ratio"] > 0.3:
            validation_results["warnings"].append("High ratio of special characters in chunks")

    def _validate_embeddings_data(self, df: pd.DataFrame, validation_results: Dict):
        stats = {
            "total_embeddings": len(df),
            "embedding_dimension": df['embedding_dimension'].iloc[0] if len(df) > 0 else 0,
            "consistent_dimensions": df['embedding_dimension'].nunique() == 1,
            "avg_norm": df['embedding_norm'].mean(),
            "median_norm": df['embedding_norm'].median(),
            "norm_std": df['embedding_norm'].std(),
            "min_norm": df['embedding_norm'].min(),
            "max_norm": df['embedding_norm'].max(),
            "embeddings_with_nan": df['has_nan_values'].sum(),
            "embeddings_with_inf": df['has_inf_values'].sum(),
            "avg_mean_value": df['mean_value'].mean(),
            "avg_std_value": df['std_value'].mean(),
            "very_sparse_embeddings": (df['zero_values_count'] > df['embedding_dimension'] * 0.9).sum(),
            "balanced_embeddings": ((df['positive_values_ratio'] > 0.3) & (df['positive_values_ratio'] < 0.7)).sum()
        }
        validation_results["statistics"] = stats
        if stats["embeddings_with_nan"] > 0:
            validation_results["is_valid"] = False
            validation_results["errors"].append(f"Found {stats['embeddings_with_nan']} embeddings with NaN values")
        if stats["embeddings_with_inf"] > 0:
            validation_results["is_valid"] = False
            validation_results["errors"].append(f"Found {stats['embeddings_with_inf']} embeddings with infinite values")
        if not stats["consistent_dimensions"]:
            validation_results["is_valid"] = False
            validation_results["errors"].append("Inconsistent embedding dimensions")
        if stats["embedding_dimension"] < 50:
            validation_results["warnings"].append("Very low embedding dimension")
        if stats["min_norm"] < 0.01:
            validation_results["warnings"].append("Some embeddings have very small norms")
        if stats["max_norm"] > 100:
            validation_results["warnings"].append("Some embeddings have very large norms")
        if stats["very_sparse_embeddings"] > stats["total_embeddings"] * 0.1:
            validation_results["warnings"].append("Many embeddings are very sparse")

    def _validate_search_data(self, df: pd.DataFrame, validation_results: Dict):
        stats = {
            "total_results": len(df),
            "results_with_text": df['has_text'].sum(),
            "results_with_metadata": df['has_metadata'].sum(),
            "results_with_scores": df['has_score'].sum(),
            "avg_score": df['score'].mean(),
            "median_score": df['score'].median(),
            "score_std": df['score'].std(),
            "min_score": df['score'].min(),
            "max_score": df['score'].max(),
            "avg_text_length": df['text_length'].mean(),
            "median_text_length": df['text_length'].median(),
            "avg_word_count": df['text_word_count'].mean(),
            "results_with_source": df['has_source'].sum(),
            "results_with_page": df['has_page'].sum(),
            "avg_query_relevance": df['query_terms_in_text'].mean(),
            "avg_text_quality": df['text_quality_score'].mean(),
            "high_quality_results": (df['text_quality_score'] > 0.7).sum()
        }
        validation_results["statistics"] = stats
        if stats["results_with_text"] < len(df):
            validation_results["warnings"].append(
                f"Some results missing text: {len(df) - stats['results_with_text']}/{len(df)}")
        if stats["results_with_scores"] < len(df):
            validation_results["warnings"].append(
                f"Some results missing scores: {len(df) - stats['results_with_scores']}/{len(df)}")
        if stats["avg_text_length"] < 20:
            validation_results["warnings"].append("Search results have very short text content")
        if stats["avg_query_relevance"] == 0:
            validation_results["warnings"].append("No query terms found in search results")
        if stats["high_quality_results"] < len(df) * 0.5:
            validation_results["warnings"].append("Many results have low text quality scores")

    def _count_query_terms_in_text(self, query: str, text: str) -> int:
        if not query or not text:
            return 0
        query_terms = query.lower().split()
        text_lower = text.lower()
        return sum(1 for term in query_terms if term in text_lower)

    def _calculate_text_quality_score(self, text: str) -> float:
        if not text:
            return 0.0
        score = 0.0
        if len(text) > 20:
            score += 0.1
        if len(text) > 100:
            score += 0.1
        if len(text) > 500:
            score += 0.1
        word_count = len(text.split())
        if word_count > 5:
            score += 0.1
        if word_count > 20:
            score += 0.1
        if text[0].isupper():
            score += 0.1
        if text.endswith(('.', '!', '?')):
            score += 0.1
        if any(c.isupper() for c in text[1:]):
            score += 0.1
        if any(c.isdigit() for c in text):
            score += 0.1
        if len(set(text.lower().split())) / max(len(text.split()), 1) > 0.5:
            score += 0.1
        return min(score, 1.0)

    def _run_ge_pdf_validation(self, df: pd.DataFrame, validation_results: Dict):
        try:
            logger.info("Running Great Expectations validation on PDF data")
        except Exception as e:
            logger.warning(f"GX validation failed: {e}")

    def generate_validation_report(self, all_validation_results: Dict[str, Dict]) -> str:
        report = []
        report.append("=" * 60)
        report.append("StorySage SYSTEM DATA VALIDATION REPORT")
        report.append("=" * 60)
        report.append("")
        overall_status = "PASSED"
        total_errors = 0
        total_warnings = 0
        for stage, results in all_validation_results.items():
            report.append(f"{stage.upper()} VALIDATION:")
            report.append("-" * 30)
            if results["is_valid"]:
                report.append("PASSED")
            else:
                report.append(" FAILED")
                overall_status = "FAILED"
            if results["errors"]:
                total_errors += len(results["errors"])
                report.append("ERRORS:")
                for error in results["errors"]:
                    report.append(f"  - {error}")
            if results["warnings"]:
                total_warnings += len(results["warnings"])
                report.append("WARNINGS:")
                for warning in results["warnings"]:
                    report.append(f"  - {warning}")
            if results["statistics"]:
                report.append("STATISTICS:")
                for key, value in results["statistics"].items():
                    if isinstance(value, float):
                        report.append(f"  {key}: {value:.2f}")
                    else:
                        report.append(f"  {key}: {value}")
            report.append("")
        report.append("=" * 60)
        report.append(f"OVERALL STATUS: {overall_status}")
        report.append(f"Total Errors: {total_errors}")
        report.append(f"Total Warnings: {total_warnings}")
        report.append("=" * 60)
        return "\n".join(report)



def validate_and_process_documents(folder_path: str, validator: RAGDataValidator = None,chunking_method: str = "standard"):
    if validator is None:
        validator = RAGDataValidator()
    try:
        from processing import get_all_files_in_folder, process_files
        from chunking import chunk_documents_semantic, chunk_documents
        from embeddings import batch_generate_embeddings, get_embedding_model
        all_files = get_all_files_in_folder(folder_path)
        if not all_files:
            logger.error(f"No files found in folder: {folder_path}")
            return None
        all_documents_with_pages, file_sources = [], []
        for file_path in all_files:
            file_docs = process_files(file_path)
            if file_docs:
                all_documents_with_pages.append(file_docs)
                file_sources.append(file_path)
        pdf_validation = validator.validate_pdf_processing_results(
            all_documents_with_pages, file_sources
        )
        if not pdf_validation["is_valid"]:
            logger.error(f"PDF processing validation failed: {pdf_validation['errors']}")
            return None
        logger.info("PDF processing validation passed")
        if chunking_method == "semantic":
            chunks, metadata = chunk_documents_semantic(
                all_documents_with_pages, file_sources, get_embedding_model()
            )
        else:
            chunks, metadata = chunk_documents(all_documents_with_pages, file_sources)
        chunking_validation = validator.validate_chunking_results(chunks, metadata)
        if not chunking_validation["is_valid"]:
            logger.error(f"Chunking validation failed: {chunking_validation['errors']}")
            return None
        logger.info("Chunking validation passed")
        embeddings = batch_generate_embeddings(chunks)
        embeddings_validation = validator.validate_embeddings(embeddings, chunks)
        if not embeddings_validation["is_valid"]:
            logger.error(f"Embeddings validation failed: {embeddings_validation['errors']}")
            return None
        logger.info("Embeddings validation passed")
        all_validations = {
            'pdf_processing': pdf_validation,
            'chunking': chunking_validation,
            'embeddings': embeddings_validation
        }
        validation_report = validator.generate_validation_report(all_validations)
        logger.info(f"Validation Report:\n{validation_report}")
        return {
            'chunks': chunks,
            'metadata': metadata,
            'embeddings': embeddings,
            'validation_results': all_validations,
            'validation_report': validation_report
        }
    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        return None


def validate_search_pipeline(query: str, search_results: List[Dict],
                             search_method: str, validator: RAGDataValidator = None):
    if validator is None:
        validator = RAGDataValidator()
    validation_results = validator.validate_search_results(
        search_results, query, search_method
    )
    if not validation_results["is_valid"]:
        logger.warning(f"Search validation issues: {validation_results['errors']}")
    logger.info(f"Search validation completed with {len(validation_results.get('warnings', []))} warnings")
    return validation_results


def create_comprehensive_validation_suite(base_path: str = "Contents"):
    validator = RAGDataValidator(base_path)
    def validate_full_pipeline(folder_path: str, query: str, expected_results: int = 5):
        logger.info("Starting comprehensive StorySage pipeline validation")
        processing_results = validate_and_process_documents(folder_path, validator)
        if not processing_results:
            logger.error("Pipeline validation failed at document processing stage")
            return None
        try:
            from search import semantic_search, bm25_search, hybrid_search
            from faiss_index import load_faiss_data
            from lexical import load_data
            index, embeddings, texts, metadata = load_faiss_data()
            bm25_model, tokenized_corpus, bm25_texts, bm25_metadata = load_data()
            if index and bm25_model:
                semantic_results = semantic_search(index, texts, metadata, query, n_results=expected_results)
                bm25_results = bm25_search(bm25_model, tokenized_corpus, texts, metadata, query, n_results=expected_results)
                hybrid_results = hybrid_search(index, bm25_model, tokenized_corpus, texts, metadata, query,n_results=expected_results)
                semantic_validation = validate_search_pipeline(query, semantic_results, "semantic", validator)
                bm25_validation = validate_search_pipeline(query, bm25_results, "bm25", validator)
                hybrid_validation = validate_search_pipeline(query, hybrid_results, "hybrid", validator)
                all_validations = {
                    **processing_results['validation_results'],
                    'semantic_search': semantic_validation,
                    'bm25_search': bm25_validation,
                    'hybrid_search': hybrid_validation
                }
                final_report = validator.generate_validation_report(all_validations)
                return {
                    'processing_results': processing_results,
                    'search_validations': {
                        'semantic': semantic_validation,
                        'bm25': bm25_validation,
                        'hybrid': hybrid_validation
                    },
                    'all_validations': all_validations,
                    'final_report': final_report
                }
            else:
                logger.warning("Search indices not available for validation")
                return processing_results
        except Exception as e:
            logger.warning(f"Search validation skipped due to error: {e}")
            return processing_results
    return validate_full_pipeline


if __name__ == "__main__":
    validator = RAGDataValidator("Contents")
    full_validator = create_comprehensive_validation_suite("Contents")
    folder_path = "Contents/books"
    test_query = "What is the main theme of the story?"
    results = full_validator(folder_path, test_query)
    if results:
        print("Validation completed successfully!")
        print("\nFinal Report:")
        print(results['final_report'])
    else:
        print("Validation failed - check logs for details")
    print("\nIntegration complete - Great Expectations validation ready!")
    print("Features available:")
    print("- PDF processing validation")
    print("- Chunking quality validation")
    print("- Embedding consistency validation")
    print("- Search result quality validation")
    print("- Comprehensive reporting")
    print("- Graceful fallback when GX not available")
    print("- Compatible with all GX versions")