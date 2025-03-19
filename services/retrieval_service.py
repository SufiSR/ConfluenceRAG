import logging
import re
from typing import Dict, List, Tuple, Optional

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from configuration import get_embeddings_model, get_llm
from utilities.vector_store import VectorStore
from utilities.confluence_connector import ConfluenceConnector
import json

logger = logging.getLogger(__name__)


class RetrievalService:
    def __init__(self, profile="internal-confluence"):
        self.vector_store = VectorStore(profile)
        self.embeddings = get_embeddings_model(profile)
        self.llm = get_llm(profile)
        self.confluence = ConfluenceConnector(profile)
        self.profile = profile

        # Default prompt
        self.prompt_template = (
            "You are a helpful assistant. Answer the question using the provided context. "
            "Generate your final response after adjusting it to increase accuracy and relevance. "
            "Now only show your final response! Do not provide any explanations or details. "
            "Please do not make up any information - stay strict on the context. "
            "Also, indicate the level of confidence you have that the answer is useful to the user in %."
        )

    def retrieve_context(self, query: str, k: int = 4) -> List[Document]:
        """Retrieve relevant context for a query."""
        try:
            return self.vector_store.similarity_search(query, k=k)
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return []

    def get_effective_page_id(self, doc: Document) -> Optional[str]:
        """Get the effective page ID from a document, handling both pages and attachments."""
        metadata = doc.metadata
        if metadata.get('is_attachment', False):
            return metadata.get('parent_page_id')
        else:
            return metadata.get('page_id')

    # In retrieval_service.py, update the get_additional_context method

    def get_additional_context(self, documents: List[Document]) -> List[Document]:
        """Get additional context from parent and child pages."""
        additional_docs = []

        # Track page IDs to avoid duplicates
        processed_page_ids = set()

        for doc in documents:
            # Get the effective page ID (either page_id or parent_page_id for attachments)
            page_id = self.get_effective_page_id(doc)

            if not page_id or page_id in processed_page_ids:
                continue

            processed_page_ids.add(page_id)

            # Get parent pages (using hierarchy information)
            hierarchy_text = doc.metadata.get('hierarchy_text', '')
            hierarchy_json = doc.metadata.get('hierarchy_json', '')

            hierarchy = []
            if hierarchy_json:
                try:
                    hierarchy = json.loads(hierarchy_json)
                except json.JSONDecodeError:
                    # Fallback to splitting the text
                    hierarchy = hierarchy_text.split(' > ') if hierarchy_text else []
            elif hierarchy_text:
                hierarchy = hierarchy_text.split(' > ')

            if hierarchy and len(hierarchy) > 0:
                for i in range(len(hierarchy)):
                    parent_title = hierarchy[i]
                    # This is a simplified approach - in a real system, you would need
                    # to map titles to page IDs more accurately
                    parent_docs = self.vector_store.db.similarity_search(parent_title, k=2)
                    for parent_doc in parent_docs:
                        if parent_doc.metadata.get('title') == parent_title:
                            additional_docs.append(parent_doc)

            # Get child pages
            try:
                child_pages = self.confluence.get_child_pages(page_id)
                for child_page in child_pages:
                    child_id = child_page.get('id')
                    if child_id and child_id not in processed_page_ids:
                        child_docs = self.vector_store.get_documents_by_metadata('page_id', child_id)
                        if child_docs:
                            # Add only one representative chunk per child page
                            additional_docs.append(child_docs[0])
                            processed_page_ids.add(child_id)
            except Exception as e:
                logger.error(f"Error getting child pages for {page_id}: {str(e)}")

        return additional_docs

    def extract_confidence(self, text: str) -> Tuple[str, float]:
        """Extract the confidence percentage from the LLM response."""
        # Pattern to match confidence score in the hidden format
        pattern = r'CONFIDENCE_SCORE:\s*(\d+(?:\.\d+)?)'
        match = re.search(pattern, text)

        if match:
            confidence = float(match.group(1))
            # Remove the confidence statement from the text
            clean_text = re.sub(r'\s*CONFIDENCE_SCORE:\s*\d+(?:\.\d+)?\s*$', '', text).strip()
            return clean_text, confidence

        # Fallback pattern for other confidence formats
        alt_pattern = r'Confidence(?:\s+in\s+the\s+usefulness\s+of\s+this\s+answer)?:\s*(\d+(?:\.\d+)?)%'
        alt_match = re.search(alt_pattern, text)

        if alt_match:
            confidence = float(alt_match.group(1))
            # Remove the confidence statement from the text
            clean_text = re.sub(
                r'\s*Confidence(?:\s+in\s+the\s+usefulness\s+of\s+this\s+answer)?:\s*\d+(?:\.\d+)?%\s*$', '',
                text).strip()
            return clean_text, confidence

        return text, 0.0

    def answer_query(self, query: str, confidence_threshold: float = 90.1) -> Dict:
        """Answer a query using RAG approach with confidence assessment."""
        logger.info(f"Processing query: {query}")

        # Retrieve initial context
        initial_docs = self.retrieve_context(query)

        if not initial_docs:
            return {
                "answer": "I'm sorry, but I couldn't find any relevant information to answer your question.",
                "confidence": 0.0,
                "sources": []
            }

        # Prepare context text
        context = "\n\n".join(
            [f"Source: {doc.metadata.get('title', 'Unknown')}\n{doc.page_content}" for doc in initial_docs])

        # Modify prompt to instruct not to include confidence in the answer text
        modified_prompt = (
            "You are a helpful assistant. Answer the question using the provided context. "
            "Generate your final response after adjusting it to increase accuracy and relevance. "
            "Do not provide any explanations about your process. "
            "Please do not make up any information - stay strict on the context. "
            "DO NOT include any statement about confidence or usefulness in your answer. "
            "After you've completed your answer, on a new line that will not be shown to the user, "
            "provide your confidence score in exactly this format: CONFIDENCE_SCORE: X"
        )

        # Get initial answer
        messages = [
            SystemMessage(content=modified_prompt),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}")
        ]

        response = self.llm.invoke(messages)
        answer_text = response.content

        # Extract confidence
        clean_answer, confidence = self.extract_confidence(answer_text)

        # If confidence is below threshold, get additional context
        if confidence < confidence_threshold:
            logger.info(f"Initial confidence ({confidence}%) below threshold, retrieving additional context")

            additional_docs = self.get_additional_context(initial_docs)

            if additional_docs:
                # Combine initial and additional context
                all_docs = initial_docs + additional_docs

                # Prepare expanded context
                expanded_context = "\n\n".join(
                    [f"Source: {doc.metadata.get('title', 'Unknown')}\n{doc.page_content}" for doc in all_docs])

                # Get improved answer
                expanded_messages = [
                    SystemMessage(content=modified_prompt),
                    HumanMessage(content=f"Context:\n{expanded_context}\n\nQuestion: {query}")
                ]

                expanded_response = self.llm.invoke(expanded_messages)
                expanded_answer_text = expanded_response.content

                # Extract confidence from expanded answer
                clean_answer, expanded_confidence = self.extract_confidence(expanded_answer_text)

                # Use expanded answer if confidence improved
                if expanded_confidence > confidence:
                    confidence = expanded_confidence
                    all_docs = initial_docs + additional_docs
                else:
                    all_docs = initial_docs
            else:
                all_docs = initial_docs
        else:
            all_docs = initial_docs

        # Prepare sources information
        sources = []
        for doc in all_docs:
            # Get appropriate source link based on whether it's an attachment or page
            link = doc.metadata.get("link", "")

            source = {
                "title": doc.metadata.get("title", "Unknown"),
                "link": link,
                "space": doc.metadata.get("space", ""),
                "is_attachment": doc.metadata.get("is_attachment", False),
            }
            if source not in sources:  # Avoid duplicates
                sources.append(source)

        return {
            "answer": clean_answer,
            "confidence": confidence,
            "sources": sources
        }
