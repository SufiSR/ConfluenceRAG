import argparse
import logging
import sys

# Import necessary functions and classes for configuration and processing.
from configuration import get_confluence_spaces
from utilities.confluence_connector import ConfluenceConnector
from utilities.data_processor import DataProcessor
from utilities.vector_store import VectorStore

# Configure logging to output messages to both a file and the console.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/rag_setup.log"),  # Logs are saved to a file for persistent record.
        logging.StreamHandler()  # Logs are also printed to the console.
    ]
)

# Create a logger instance for this module.
logger = logging.getLogger(__name__)


def initial_setup(profile="internal-confluence"):
    """
    Perform initial setup by indexing all content from configured Confluence spaces.

    This function retrieves the list of spaces based on the given profile, connects to Confluence,
    processes the retrieved content, and indexes the processed documents into a vector store.

    Args:
        profile (str): The configuration profile to use. Default is "internal-confluence".

    Returns:
        None
    """
    logger.info(f"Starting initial setup for profile: {profile}")

    # Retrieve the list of Confluence spaces configured for the given profile.
    spaces = get_confluence_spaces(profile)
    logger.info(f"Using spaces: {', '.join(spaces)} for profile: {profile}")

    # Initialize the necessary components with the specified profile.
    confluence = ConfluenceConnector(profile)  # Connector to interact with Confluence API.
    processor = DataProcessor()  # Component to process and transform the raw content.
    vector_store = VectorStore(profile)  # Vector store for indexing and searching processed documents.

    # Retrieve all content from the configured Confluence spaces.
    content = confluence.get_all_content_from_spaces(spaces)
    logger.info(f"Retrieved {len(content)} content items from Confluence")

    # Process the retrieved content to prepare it for indexing.
    documents = processor.process_content(content)

    # Add the processed documents to the vector store for later retrieval.
    vector_store.add_documents(documents)

    logger.info(f"Initial setup completed for profile: {profile}")


if __name__ == "__main__":
    # Set up command-line argument parsing.
    parser = argparse.ArgumentParser(description="Setup Confluence RAG")
    parser.add_argument(
        '--profile',
        type=str,
        default="internal-confluence",
        help='Configuration profile to use (internal-confluence or online-help)'
    )

    # Parse the command-line arguments.
    args = parser.parse_args()

    try:
        # Run the initial setup process using the provided profile.
        initial_setup(args.profile)
        # Exit with status code 0 indicating success.
        sys.exit(0)
    except Exception as e:
        # Log any errors encountered during setup with full traceback details.
        logger.error(f"Error during setup for profile {args.profile}: {str(e)}", exc_info=True)
        # Exit with a non-zero status code to indicate failure.
        sys.exit(1)
