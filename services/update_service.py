import logging
import os
from datetime import datetime, timedelta
from typing import List, Set

from configuration import get_confluence_spaces
from utilities.confluence_connector import ConfluenceConnector
from utilities.data_processor import DataProcessor
from utilities.vector_store import VectorStore

logger = logging.getLogger(__name__)


class UpdateService:
    def __init__(self, profile="internal-confluence"):
        self.confluence = ConfluenceConnector(profile)
        self.processor = DataProcessor()
        self.vector_store = VectorStore(profile)
        self.spaces = get_confluence_spaces(profile)
        self.profile = profile

    def get_current_page_ids(self, spaces: List[str]) -> Set[str]:
        """Get all current page IDs from specified spaces."""
        page_ids = set()
        for space in spaces:
            pages = self.confluence.get_pages_from_space(space)
            for page in pages:
                page_ids.add(page.get('id', ''))
        return page_ids

    def get_stored_page_ids(self) -> Set[str]:
        """Get all page IDs currently stored in the vector store."""
        stored_ids = set()
        results = self.vector_store.db.get()
        if results and 'metadatas' in results:
            for metadata in results['metadatas']:
                if 'page_id' in metadata:
                    stored_ids.add(metadata['page_id'])
        return stored_ids

    def get_stored_page_ids_by_space(self, space: str) -> Set[str]:
        """
        Retrieve stored page IDs from the vector store for a given space.
        Assumes that each stored document's metadata includes a 'space' field.
        """
        stored_ids = set()
        results = self.vector_store.db.get()
        if results and 'metadatas' in results:
            for metadata in results['metadatas']:
                if metadata.get('space', '') == space and 'page_id' in metadata:
                    stored_ids.add(metadata['page_id'])
        return stored_ids

    def update(self) -> None:
        """Existing update process."""
        logger.info(f"Starting update process for profile: {self.profile}")

        # Get current page IDs from Confluence
        current_page_ids = self.get_current_page_ids(self.spaces)
        logger.info(f"Found {len(current_page_ids)} pages in Confluence")

        # Get stored page IDs from vector store
        stored_page_ids = self.get_stored_page_ids()
        logger.info(f"Found {len(stored_page_ids)} pages in vector store")

        # Identify new, deleted, and potentially modified pages
        new_page_ids = current_page_ids - stored_page_ids
        deleted_page_ids = stored_page_ids - current_page_ids
        existing_page_ids = current_page_ids & stored_page_ids

        logger.info(f"New pages: {len(new_page_ids)}")
        logger.info(f"Deleted pages: {len(deleted_page_ids)}")
        logger.info(f"Existing pages to check for modifications: {len(existing_page_ids)}")

        # Process deleted pages
        for page_id in deleted_page_ids:
            self.vector_store.delete_by_metadata_field('page_id', page_id)

        # Process new pages
        new_content = []
        for page_id in new_page_ids:
            text, metadata = self.confluence.get_page_content(page_id)
            hierarchy = self.confluence.get_page_hierarchy(page_id)
            metadata['hierarchy'] = hierarchy
            new_content.append((text, metadata))

            attachments = self.confluence.get_attachments(page_id)
            attachment_list = []
            if isinstance(attachments, dict) and 'results' in attachments:
                attachment_list = attachments['results']
            elif isinstance(attachments, list):
                attachment_list = attachments

            for attachment in attachment_list:
                filename = attachment.get('title', '')
                file_extension = os.path.splitext(filename)[1].lower()
                if file_extension not in ['.pdf', '.docx', '.pptx']:
                    logger.info(f"Skipping unsupported attachment: {filename}")
                    continue

                attachment_text, attachment_metadata = self.confluence.get_attachment_content(attachment, page_id)
                if attachment_text:
                    attachment_metadata['hierarchy'] = hierarchy + [metadata.get('title', '')]
                    new_content.append((attachment_text, attachment_metadata))

        if new_content:
            new_documents = self.processor.process_content(new_content)
            self.vector_store.add_documents(new_documents)

        # Check for modifications in existing pages
        modified_content = []
        for page_id in existing_page_ids:
            text, metadata = self.confluence.get_page_content(page_id)
            last_modified = metadata.get('last_modified', '')

            stored_docs = self.vector_store.get_documents_by_metadata('page_id', page_id)
            if stored_docs:
                stored_last_modified = stored_docs[0].metadata.get('last_modified', '')
                if last_modified != stored_last_modified:
                    logger.info(f"Page {page_id} was modified, updating")
                    self.vector_store.delete_by_metadata_field('page_id', page_id)
                    self.vector_store.delete_by_metadata_field('parent_page_id', page_id)

                    hierarchy = self.confluence.get_page_hierarchy(page_id)
                    metadata['hierarchy'] = hierarchy
                    modified_content.append((text, metadata))

                    attachments = self.confluence.get_attachments(page_id)
                    attachment_list = []
                    if isinstance(attachments, dict) and 'results' in attachments:
                        attachment_list = attachments['results']
                    elif isinstance(attachments, list):
                        attachment_list = attachments

                    for attachment in attachment_list:
                        filename = attachment.get('title', '')
                        file_extension = os.path.splitext(filename)[1].lower()
                        if file_extension not in ['.pdf', '.docx', '.pptx']:
                            logger.info(f"Skipping unsupported attachment: {filename}")
                            continue

                        attachment_text, attachment_metadata = self.confluence.get_attachment_content(attachment, page_id)
                        if attachment_text:
                            attachment_metadata['hierarchy'] = hierarchy + [metadata.get('title', '')]
                            modified_content.append((attachment_text, attachment_metadata))

        if modified_content:
            modified_documents = self.processor.process_content(modified_content)
            self.vector_store.add_documents(modified_documents)

        logger.info(f"Update process completed for profile: {self.profile}")

    def update_efficient(self) -> None:
        """
        Efficient update process:
         - For deletion: For each space, retrieve the minimal list of page IDs from Confluence,
           and compare them to the stored page IDs in the vector store.
         - For new/modified pages: For each space, retrieve pages modified in the last two weeks.
           Compare their 'last_modified' timestamps to determine if they are new or updated.
        """
        logger.info(f"Starting efficient update process for profile: {self.profile}")

        # --- Deletion Phase (Space by Space) ---
        for space in self.spaces:
            confluence_page_ids = self.confluence.get_page_ids_from_space_minimal(space)
            stored_page_ids = self.get_stored_page_ids_by_space(space)
            deleted_page_ids = stored_page_ids - confluence_page_ids
            logger.info(f"Space {space}: Found {len(deleted_page_ids)} deleted pages")
            for page_id in deleted_page_ids:
                self.vector_store.delete_by_metadata_field('page_id', page_id)

        # --- New/Modified Pages Phase ---
        modified_since_date = (datetime.now() - timedelta(weeks=2)).isoformat()
        new_and_modified_content = []
        for space in self.spaces:
            modified_pages = self.confluence.get_pages_modified_since(space, modified_since_date)
            logger.info(f"Space {space}: Found {len(modified_pages)} pages modified since {modified_since_date}")
            for page in modified_pages:
                page_id = page.get('id', '')
                text, metadata = self.confluence.get_page_content(page_id)
                hierarchy = self.confluence.get_page_hierarchy(page_id)
                metadata['hierarchy'] = hierarchy

                stored_docs = self.vector_store.get_documents_by_metadata('page_id', page_id)
                if stored_docs:
                    stored_last_modified = stored_docs[0].metadata.get('last_modified', '')
                    if metadata.get('last_modified', '') != stored_last_modified:
                        logger.info(f"Page {page_id} was modified, updating")
                        self.vector_store.delete_by_metadata_field('page_id', page_id)
                        self.vector_store.delete_by_metadata_field('parent_page_id', page_id)
                        new_and_modified_content.append((text, metadata))

                        attachments = self.confluence.get_attachments(page_id)
                        attachment_list = []
                        if isinstance(attachments, dict) and 'results' in attachments:
                            attachment_list = attachments['results']
                        elif isinstance(attachments, list):
                            attachment_list = attachments

                        for attachment in attachment_list:
                            filename = attachment.get('title', '')
                            file_extension = os.path.splitext(filename)[1].lower()
                            if file_extension not in ['.pdf', '.docx', '.pptx']:
                                logger.info(f"Skipping unsupported attachment: {filename}")
                                continue
                            attachment_text, attachment_metadata = self.confluence.get_attachment_content(attachment, page_id)
                            if attachment_text:
                                attachment_metadata['hierarchy'] = hierarchy + [metadata.get('title', '')]
                                new_and_modified_content.append((attachment_text, attachment_metadata))
                else:
                    logger.info(f"Page {page_id} is new, adding")
                    new_and_modified_content.append((text, metadata))
                    attachments = self.confluence.get_attachments(page_id)
                    attachment_list = []
                    if isinstance(attachments, dict) and 'results' in attachments:
                        attachment_list = attachments['results']
                    elif isinstance(attachments, list):
                        attachment_list = attachments

                    for attachment in attachment_list:
                        filename = attachment.get('title', '')
                        file_extension = os.path.splitext(filename)[1].lower()
                        if file_extension not in ['.pdf', '.docx', '.pptx']:
                            logger.info(f"Skipping unsupported attachment: {filename}")
                            continue
                        attachment_text, attachment_metadata = self.confluence.get_attachment_content(attachment, page_id)
                        if attachment_text:
                            attachment_metadata['hierarchy'] = hierarchy + [metadata.get('title', '')]
                            new_and_modified_content.append((attachment_text, attachment_metadata))

        if new_and_modified_content:
            processed_documents = self.processor.process_content(new_and_modified_content)
            self.vector_store.add_documents(processed_documents)

        logger.info(f"Efficient update process completed for profile: {self.profile}")

    def run_scheduled_update(self) -> None:
        """Run the update process based on schedule."""
        try:
            logger.info(f"Running scheduled update at {datetime.now()} for profile: {self.profile}")
            # You can choose to call the efficient update instead of the full update
            self.update_efficient()
        except Exception as e:
            logger.error(f"Error during scheduled update for profile {self.profile}: {str(e)}")
