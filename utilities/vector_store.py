import logging
import os
from typing import List

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from configuration import get_embeddings_model, get_chroma_directory

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, profile="internal-confluence"):
        self.embeddings = get_embeddings_model(profile)
        self.persist_directory = get_chroma_directory(profile)
        self.profile = profile

        # Create directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)

        # Initialize Chroma client with custom settings
        chroma_client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False
            )
        )

        # Initialize Chroma with the custom client
        self.db = Chroma(
            client=chroma_client,
            embedding_function=self.embeddings
        )

    def add_documents(self, documents: List[Document], batch_size: int = 5000) -> None:
        """
        Add documents to the vector store in batches to avoid exceeding maximum batch size.

        Args:
            documents: List of documents to add
            batch_size: Maximum number of documents to add in a single batch
        """
        try:
            total_docs = len(documents)
            logger.info(f"Adding {total_docs} documents to the vector store in batches of {batch_size}")

            # Process documents in batches
            for i in range(0, total_docs, batch_size):
                end_idx = min(i + batch_size, total_docs)
                batch = documents[i:end_idx]

                logger.info(
                    f"Adding batch {i // batch_size + 1}/{(total_docs + batch_size - 1) // batch_size}: documents {i} to {end_idx - 1}")
                self.db.add_documents(batch)

            logger.info(f"Successfully added all {total_docs} documents to the vector store")
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")

    def delete_by_metadata_field(self, field_name: str, field_value: str) -> None:
        """Delete documents with matching metadata field."""
        try:
            ids_to_delete = []

            # Get all documents with matching field
            results = self.db.get(where={field_name: field_value})

            if results and 'ids' in results:
                ids_to_delete.extend(results['ids'])

            if ids_to_delete:
                self.db.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} documents with {field_name}={field_value}")
            else:
                logger.info(f"No documents found with {field_name}={field_value}")

        except Exception as e:
            logger.error(f"Error deleting documents from vector store: {str(e)}")

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search for similar documents based on query."""
        try:
            return self.db.similarity_search(query, k=k)
        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}")
            return []

    def similarity_search_by_vector(self, embedding: List[float], k: int = 4) -> List[Document]:
        """Search for similar documents based on embedding vector."""
        try:
            return self.db.similarity_search_by_vector(embedding, k=k)
        except Exception as e:
            logger.error(f"Error during similarity search by vector: {str(e)}")
            return []

    def get_documents_by_metadata(self, metadata_field: str, metadata_value: str) -> List[Document]:
        """Get documents with matching metadata field."""
        try:
            results = self.db.get(where={metadata_field: metadata_value})

            if not results or 'documents' not in results or 'metadatas' not in results:
                return []

            documents = []
            for i, doc_text in enumerate(results['documents']):
                metadata = results['metadatas'][i] if i < len(results['metadatas']) else {}
                documents.append(Document(page_content=doc_text, metadata=metadata))

            return documents

        except Exception as e:
            logger.error(f"Error getting documents by metadata: {str(e)}")
            return []
