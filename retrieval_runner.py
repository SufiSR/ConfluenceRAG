import logging
from services.retrieval_service import RetrievalService

# Configure logging: set the logging level, message format, and output handlers (file and console).
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/rag_retrieval.log"),  # Save logs to a file.
        logging.StreamHandler()  # Also output logs to the console.
    ]
)

# Create a logger instance for this module.
logger = logging.getLogger(__name__)


def get_formatted_answer(query: str, profile="internal-confluence", confidence_threshold: float = 89) -> str:
    """
    Retrieve and format an answer based on a query using the retrieval service.

    This function initializes the RetrievalService with the given profile,
    sends the query along with a confidence threshold to get a response,
    and then formats the returned answer and its sources.

    Args:
        query (str): The query string for which an answer is requested.
        profile (str, optional): The configuration profile to use (e.g., "internal-confluence").
                                 Defaults to "internal-confluence".
        confidence_threshold (float, optional): The minimum confidence level required for the answer.
                                                Defaults to 89.

    Returns:
        str: The formatted answer including the answer text and a list of sources (if available).
    """
    # Initialize the retrieval service with the specified profile.
    retrieval_service = RetrievalService(profile)

    # Send the query to the retrieval service with the specified confidence threshold.
    result = retrieval_service.answer_query(query, confidence_threshold=confidence_threshold)

    # Extract the confidence level and answer text from the result.
    confidence = result['confidence']
    answer = result['answer']

    # Begin building the formatted result with the answer text.
    formatted_result = answer + "\n"

    # If the result includes any sources, append them to the formatted result.
    if result['sources']:
        formatted_result += "\nSOURCES:\n"
        formatted_result += "-" * 80 + "\n"

        # Loop over each source and format its details.
        for i, source in enumerate(result['sources'], 1):
            title = source['title']
            space = source['space']  # Note: The 'space' is retrieved but not displayed.
            link = source['link']
            # Determine if the source is an attachment or a page.
            source_type = "Attachment" if source.get('is_attachment') else "Page"

            # Append the formatted source information.
            formatted_result += f"{i}. {title} ({source_type}):\n"
            if link:
                formatted_result += f"{link}\n"

    # Print the final confidence level to the console for debugging purposes.
    print(f"Final confidence level: {confidence}")

    return formatted_result


# Example usage: run the function when this script is executed directly.
if __name__ == "__main__":
    query = "What can you tell me about Version 10.12.1"
    # Get the formatted answer using the "internal-confluence" profile.
    formatted_answer = get_formatted_answer(query, "internal-confluence")
    # Print the formatted answer to the console.
    print(formatted_answer)
