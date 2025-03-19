# Confluence RAG Integration

This project implements a Retrieval-Augmented Generation (RAG) system that integrates Confluence content with a vector store to provide real-time chat completions and question-answering functionality. The system includes:

- **Chat Completions API:** A FastAPI endpoint that streams responses as server-sent events.
- **Update Service:** A script that synchronizes and updates Confluence content.
- **Initial Setup:** A script to index all content from configured Confluence spaces.
- **Retrieval Service:** A module that queries the vector store and formats answers along with source details.

## Features

- **Real-time Chat Responses:** Stream chat completions via a FastAPI API.
- **Content Synchronization:** Update Confluence content efficiently.
- **Content Indexing:** Perform an initial index of Confluence spaces for fast retrieval.
- **Formatted Answers:** Retrieve and display answers with confidence levels and sources.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Required packages (see `requirements.txt`):
  - `fastapi`
  - `uvicorn`
  - `pydantic`
  - `python-dotenv`
  - And others as needed.

### Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/confluence-rag-integration.git
   cd confluence-rag-integration
2. **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    Configure Environment Variables: Create a .env file in the project root and add the required environment variables:

4. **Configure Environment Variables: Create a .env file in the project root and add the required environment variables:**
    ```bash
    RAG_LIBRE_KEY=your_api_key_here
    # Other variables, e.g., Confluence credentials, as needed

# Usage
## Running the Chat Completions API
Start the FastAPI server using Webserver.py
    
    
## Running the Update Service
Synchronize Confluence content by running:
```bash
    python update_service.py --profile internal-confluence
```
You can specify the profile (e.g., internal-confluence, online-help or both) using the --profile argument.

## Initial Setup
Index all Confluence content with:
```bash
python setup.py --profile internal-confluence
```
This script retrieves, processes, and indexes content from the configured Confluence spaces.

## Testing the Retrieval Service
Run the retrieval service to query and display a formatted answer:
```bash
python retrieval_service.py
```
his example query demonstrates how the system fetches an answer and displays relevant sources.

## Logging
Log files are generated in the logs/ directory:

rag_update.log: Logs from the update service.
rag_setup.log: Logs from the initial setup script.
rag_retrieval.log: Logs from the retrieval service.

# License
This project is licensed under the MIT License.
