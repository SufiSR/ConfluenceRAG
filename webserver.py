import os
import time
import json
import asyncio
import traceback
import logging
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, List
import uvicorn
from dotenv import load_dotenv
from retrieval_runner import get_formatted_answer

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)

# Retrieve API key from environment variable
API_KEY = os.getenv("RAG_LIBRE_KEY", "").strip()


def get_api_key(authorization: str = Header(None)):
    """
    Dependency function to retrieve and validate the API key from the Authorization header.

    Args:
        authorization (str): The value of the Authorization header expected to be in the form "Bearer <API_KEY>".

    Returns:
        str: The validated API key if it matches the server's configured API_KEY.

    Raises:
        HTTPException: If the header is missing, improperly formatted, or if the key does not match.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid or missing Authorization header")
    provided_key = authorization.split("Bearer ")[1].strip()
    if provided_key != API_KEY:
        logging.warning(f"Invalid API Key Attempt: {provided_key}")
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return provided_key


# Request model representing the expected request body for chat completions.
class ChatRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]


def query_rag_system(model: str, question: str) -> str:
    """
    Selects and queries the appropriate Retrieval-Augmented Generation (RAG) system based on the model provided.

    Args:
        model (str): The model identifier (e.g., "online-help" or "internal-confluence").
        question (str): The user question extracted from the chat messages.

    Returns:
        str: A formatted answer from the selected RAG system.

    Raises:
        HTTPException: If an unknown model is provided.
    """
    logging.info(f"Querying RAG system for model: {model}, question: {question}")
    if model == "online-help":
        return get_formatted_answer(question, profile="online-help")
    elif model == "internal-confluence":
        return get_formatted_answer(question, profile="internal-confluence")
    else:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model}")


async def stream_response(response_text: str, model: str):
    """
    Creates an asynchronous streaming response that yields the answer in chunks suitable for Server-Sent Events (SSE).

    This function splits the full response into lines and words, then yields each word (with appropriate spacing)
    along with line breaks, simulating a real-time streaming experience.

    Args:
        response_text (str): The complete response text to be streamed.
        model (str): The model identifier used to tag the response metadata.

    Returns:
        StreamingResponse: A FastAPI StreamingResponse that streams the JSON-formatted response chunks.
    """

    async def event_stream():
        # Split the response into lines to preserve formatting.
        lines = response_text.split("\n")
        for line in lines:
            if line:  # Process non-empty lines: split the line into words.
                words = line.split()
                for j, word in enumerate(words):
                    # Prepend a space for words after the first in a line.
                    chunk = word if j == 0 else " " + word
                    msg_id = f"pluchatcmpl-{int(time.time())}"
                    response_data = {
                        "id": msg_id,
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": chunk},
                            "finish_reason": None
                        }]
                    }
                    yield "data: " + json.dumps(response_data) + "\n\n"
                    await asyncio.sleep(0.05)  # Slight delay between each word for streaming effect.
            # After processing a line, send a newline to maintain line breaks.
            msg_id = f"pluchatcmpl-{int(time.time())}"
            newline_data = {
                "id": msg_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": "\n"},
                    "finish_reason": None
                }]
            }
            yield "data: " + json.dumps(newline_data) + "\n\n"
            await asyncio.sleep(0.05)
        # Final message indicating the completion of the response.
        msg_id = f"pluchatcmpl-{int(time.time())}"
        final_data = {
            "id": msg_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
        yield "data: " + json.dumps(final_data) + "\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/v1/chat/completions")
async def chat_completions(request: ChatRequest, api_key: str = Depends(get_api_key)):
    """
    Main API endpoint for processing chat completion requests.

    This endpoint performs the following steps:
    1. Validates the API key using the dependency.
    2. Logs the incoming message from the request.
    3. Extracts the latest user message.
    4. Calls the RAG system (via query_rag_system) to retrieve an answer.
    5. Returns the answer as a streaming response.

    Args:
        request (ChatRequest): The incoming request payload containing the model and messages.
        api_key (str): The validated API key provided via the header (ensured by the dependency).

    Returns:
        StreamingResponse: A streaming response with the RAG system's answer.

    Raises:
        HTTPException: For any server errors or issues in processing the request.
    """
    try:
        print(request.messages)  # Debug: print the list of messages
        user_message = request.messages[-1]["content"]  # Get the latest message from the conversation.
        logging.info(f"Received user message: {user_message}")
        response_text = query_rag_system(request.model, user_message)
        return await stream_response(response_text, request.model)
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


# Run the FastAPI server if this script is executed directly.
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001, log_level="info")
