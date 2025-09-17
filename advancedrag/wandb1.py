import wandb
import os
import json
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime

class WandbLogger:


    def __init__(self, project_name: str = "rag-system", entity: Optional[str] = None):

        self.project_name = project_name
        self.entity = entity
        self.run = None
        self.experiment_data = []

    def init_experiment(self, config: Dict[str, Any], experiment_name: Optional[str] = None):

        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"rag_experiment_{timestamp}"


        self.run = wandb.init(
            project=self.project_name,
            entity=self.entity,
            name=experiment_name,
            config=config,
            reinit=True
        )

        self.log_system_info()

        return self.run

    def log_system_info(self):
        if self.run is None:
            return

        system_info = {
            "python_version": __import__("sys").version,
            "available_models": self.get_available_models(),
            "dataset_info": self.get_dataset_info()
        }

        wandb.log({"system_info": system_info})

    def get_available_models(self) -> Dict[str, List[str]]:

        try:
            from embeddings import AVAILABLE_EMBEDDING_MODELS
            from reranking import get_available_reranker_models

            return {
                "embedding_models": list(AVAILABLE_EMBEDDING_MODELS.keys()),
                "reranker_models": list(get_available_reranker_models()),
                "llm_models": ["llama3", "mistral"]
            }
        except ImportError:
            return {"error": "Could not import model information"}

    def get_dataset_info(self) -> Dict[str, Any]:

        try:

            contents_path = "Contents"
            dataset_info = {}

            if os.path.exists(os.path.join(contents_path, "file.csv")):
                df = pd.read_csv(os.path.join(contents_path, "file.csv"))
                dataset_info["qa_pairs"] = len(df)
                dataset_info["columns"] = list(df.columns)


            books_path = os.path.join(contents_path, "books")
            if os.path.exists(books_path):
                pdf_files = [f for f in os.listdir(books_path) if f.lower().endswith('.pdf')]
                dataset_info["pdf_count"] = len(pdf_files)
                dataset_info["pdf_files"] = pdf_files

            return dataset_info
        except Exception as e:
            return {"error": f"Could not gather dataset info: {str(e)}"}

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):

        if self.run is None:
            print("Warning: wandb run not initialized. Call init_experiment() first.")
            return

        wandb.log(metrics, step=step)

    def log_experiment_result(self,
                              question: str,
                              response: str,
                              actual_answer: str,
                              metrics: Dict[str, float],
                              config: Dict[str, Any],
                              selected_pdf: str,
                              chunks_found: int = 0,
                              step: Optional[int] = None):
        if self.run is None:
            print("Warning: wandb run not initialized. Call init_experiment() first.")
            return


        log_data = {
            "question": question,
            "response": response,
            "actual_answer": actual_answer,
            "selected_pdf": selected_pdf,
            "chunks_found": chunks_found,
            **metrics,
            **{f"config_{k}": v for k, v in config.items()}
        }


        self.experiment_data.append(log_data.copy())

        wandb.log(log_data, step=step)

    def log_search_results(self,
                           query: str,
                           search_results: List[Dict[str, Any]],
                           search_method: str,
                           step: Optional[int] = None):

        if self.run is None:
            return

        search_table = wandb.Table(columns=["rank", "text_preview", "score", "source", "page"])

        for i, result in enumerate(search_results[:10]):  # Log top 10 results
            text_preview = result.get("text", "")[:100] + "..." if len(result.get("text", "")) > 100 else result.get(
                "text", "")
            score = result.get("score", 0.0)
            source = result.get("metadata", {}).get("source", "Unknown")
            page = result.get("metadata", {}).get("page", "Unknown")

            search_table.add_data(i + 1, text_preview, score, source, page)

        wandb.log({
            f"search_results_{search_method}": search_table,
            f"num_results_{search_method}": len(search_results),
            f"avg_score_{search_method}": sum(r.get("score", 0) for r in search_results) / len(
                search_results) if search_results else 0
        }, step=step)

    def log_confusion_matrix(self, predictions: List[str], actuals: List[str]):
        if self.run is None:
            return

        pass

    def create_summary_dashboard(self):
        if not self.experiment_data:
            print("No experiment data to summarize")
            return


        df = pd.DataFrame(self.experiment_data)

        summary_stats = {
            "avg_bert_score": df.get("bert_score", [0]).mean(),
            "avg_rouge_score": df.get("rouge_l_score", [0]).mean(),
            "avg_response_answer_bert": df.get("response_answer_bert_score", [0]).mean(),
            "best_bert_score": df.get("bert_score", [0]).max(),
            "best_rouge_score": df.get("rouge_l_score", [0]).max(),
            "total_experiments": len(df)
        }

        wandb.log({"experiment_summary": summary_stats})


        if len(df) > 0:
            summary_table = wandb.Table(dataframe=df)
            wandb.log({"all_experiments": summary_table})

    def finish_experiment(self):

        if self.run is not None:
            self.create_summary_dashboard()
            wandb.finish()
            self.run = None

    def sweep_config(self, search_space: Dict[str, Any]) -> Dict[str, Any]:

        sweep_config = {
            'method': 'grid',
            'metric': {
                'name': 'bert_score',
                'goal': 'maximize'
            },
            'parameters': search_space
        }

        return sweep_config


def create_default_search_space() -> Dict[str, Any]:

    return {
        'embedding_model': {
            'values': ['all-MiniLM-L6-v2', 'all-mpnet-base-v2']
        },
        'chunking_method': {
            'values': ['standard', 'semantic']
        },
        'search_method': {
            'values': ['semantic', 'lexical', 'hybrid']
        },
        'llm_model': {
            'values': ['llama3', 'mistral']
        },
        'use_reranker': {
            'values': [True, False]
        },
        'alpha': {
            'values': [0.3, 0.5, 0.7, 0.9]
        },
        'n_semantic': {
            'values': [3, 5, 7]
        },
        'n_lexical': {
            'values': [3, 5, 7]
        }
    }



def log_to_wandb_and_csv(wandb_logger: WandbLogger,
                         question: str,
                         response: str,
                         actual_answer: str,
                         metrics: Dict[str, float],
                         config: Dict[str, Any],
                         selected_pdf: str,
                         chunks_found: int = 0,
                         step: Optional[int] = None):

    wandb_logger.log_experiment_result(
        question=question,
        response=response,
        actual_answer=actual_answer,
        metrics=metrics,
        config=config,
        selected_pdf=selected_pdf,
        chunks_found=chunks_found,
        step=step
    )

    try:
        from common import log_max_bertscore_to_csv
        log_max_bertscore_to_csv(
            question=question,
            response=response,
            actual_answer=actual_answer,
            max_bert_score=metrics.get('bert_score', 0.0),
            response_answer_bert_score=metrics.get('response_answer_bert_score'),
            max_chunk_answer_bert_score=metrics.get('max_chunk_answer_bert_score'),
            selected_pdf=selected_pdf,
            llm_model=config.get('llm_model'),
            search_type=config.get('search_method'),
            max_rouge_score=metrics.get('rouge_l_score'),
            response_answer_rouge_score=metrics.get('response_answer_rouge_score'),
            max_chunk_answer_rouge_score=metrics.get('max_chunk_answer_rouge_score'),
            use_reranker=config.get('use_reranker'),
            reranker_model=config.get('reranker_model'),
            chunking_method=config.get('chunking_method'),
            query_optimization=config.get('use_query_optimization'),
            embedding_model=config.get('embedding_model')
        )
    except ImportError:
        print("Could not import CSV logging function")