import logging
from typing import Dict, List, Tuple
import json

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DataProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def filter_complex_metadata(self, metadata: Dict) -> Dict:
        """Filter out complex metadata types that Chroma doesn't support."""
        filtered_metadata = {}
        for key, value in metadata.items():
            # Handle lists by converting to strings
            if isinstance(value, list):
                filtered_metadata[key] = json.dumps(value)
            # Handle dictionaries by converting to strings
            elif isinstance(value, dict):
                filtered_metadata[key] = json.dumps(value)
            # Keep simple types
            elif isinstance(value, (str, int, float, bool)) or value is None:
                filtered_metadata[key] = value
            # Convert anything else to string
            else:
                filtered_metadata[key] = str(value)
        return filtered_metadata

    def process_content(self, content_list: List[Tuple[str, Dict]]) -> List[Document]:
        """Process content by chunking text and adding metadata."""
        documents = []

        for text, metadata in content_list:
            if not text:
                continue

            # Handle hierarchies as strings instead of lists
            if 'hierarchy' in metadata and isinstance(metadata['hierarchy'], list):
                # Convert hierarchy list to string
                metadata['hierarchy_text'] = ' > '.join(metadata['hierarchy'])
                # Store the original as a JSON string
                metadata['hierarchy_json'] = json.dumps(metadata['hierarchy'])
                # Remove the original list
                del metadata['hierarchy']

            # Split text into chunks
            chunks = self.text_splitter.split_text(text)

            # Create documents from chunks
            for i, chunk in enumerate(chunks):
                doc_metadata = {
                    **metadata,
                    'chunk_id': i,
                    'chunk_count': len(chunks),
                }

                # Filter complex metadata to ensure compatibility with Chroma
                doc_metadata = self.filter_complex_metadata(doc_metadata)

                doc = Document(
                    page_content=chunk,
                    metadata=doc_metadata
                )
                documents.append(doc)

        logger.info(f"Processed {len(content_list)} content items into {len(documents)} document chunks")
        return documents
