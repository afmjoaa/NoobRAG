# Following line is important even if it's not in use.
import time

from fastcoref import spacy_component
import spacy
from typing import List, Dict, Any, Optional
import logging
import torch


class CorefResolver:
    def __init__(self,
                 model_name: str = "en_core_web_trf",
                 coref_model_architecture: str = "LingMessCoref",
                 coref_model_path: str = "biu-nlp/lingmess-coref",
                 device: str = None):
        """
        Initialize the CorefResolver with specified models.

        Args:
            model_name: The SpaCy model to use.
            coref_model_architecture: The architecture for the coref model.
            coref_model_path: The path to the coref model.
            device: The device to use for computation (cpu or cuda).
                   If None, will automatically use GPU if available.
        """
        self.model_name = model_name
        self.coref_model_architecture = coref_model_architecture
        self.coref_model_path = coref_model_path

        # Auto-detect GPU if device is not specified
        if device is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logging.info(f"Automatically selected device: {self.device}")
        else:
            self.device = device

        self.nlp = None

        # Initialize the NLP pipeline
        self._initialize_pipeline()

    def _initialize_pipeline(self) -> None:
        """
        Initialize the SpaCy NLP pipeline with the FastCoref component.
        """
        try:
            self.nlp = spacy.load(self.model_name)

            # Add the FastCoref pipe with the detected or specified device
            self.nlp.add_pipe("fastcoref", config={
                'model_architecture': self.coref_model_architecture,
                'model_path': self.coref_model_path,
                'device': self.device
            })

            logging.info(f"Successfully initialized NLP pipeline with {self.model_name} and FastCoref on {self.device}")
        except Exception as e:
            logging.error(f"Failed to initialize NLP pipeline: {str(e)}")
            raise

    def _process_batch(self, texts: List[str], batch_size: int = 20) -> List[str]:
        """Process a batch of texts using spaCy's pipe with automatic batching"""
        try:
            processed_docs = self.nlp.pipe(
                texts,
                batch_size=batch_size,
                component_cfg={"fastcoref": {'resolve_text': True}}
            )
            return [doc._.resolved_text for doc in processed_docs]
        except Exception as e:
            logging.error(f"Batch processing failed: {str(e)}")
            return texts  # Return original texts on error

    def resolve_documents(self, documents: List[Dict[str, Any]], batch_size: int = 20) -> List[Dict[str, Any]]:
        """Process documents in optimized batches while preserving structure"""
        # Extract texts and original order
        text_map = [(i, doc.get('text', '')) for i, doc in enumerate(documents)]
        indices, texts = zip(*text_map) if text_map else ([], [])

        # Process in batches
        resolved_texts = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            resolved_texts.extend(self._process_batch(batch_texts, batch_size))

        # Reconstruct documents with original order and structure
        results = [doc.copy() for doc in documents]
        for idx, resolved_text in zip(indices, resolved_texts):
            if 'text' in results[idx]:
                results[idx]['text'] = resolved_text

        return results


# Example usage with large batches
if __name__ == "__main__":
    # Generate 40 batches of 20 documents each (800 total)
    sample_documents = [
        {
            "doc_id": f"<urn:uuid:{i}-{j}>",
            "text": "Angela Merkel met Barack Obama yesterday. She greeted him warmly.",
            "score": 0.823291481,
            "source": "dense"
        }
        for i in range(10)
        for j in range(20)
    ]

    try:
        resolver = CorefResolver()

        # Process all documents in optimized batches
        start_time = time.time()
        resolved_docs = resolver.resolve_documents(sample_documents, batch_size=20)

        total_time = time.time() - start_time
        print(f"Processed {len(sample_documents)} documents in {total_time:.2f} seconds")
        print(f"Throughput: {len(sample_documents) / total_time:.2f} docs/sec")

    except Exception as e:
        print(f"Error: {str(e)}")