import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from atlassian import Confluence

# Load environment variables
load_dotenv()


class ConfluenceConfig:
    def __init__(self, profile="internal-confluence"):
        suffix = f"_{profile}" if profile != "internal-confluence" else ""
        self.url = os.getenv(f"CONFLUENCE_URL{suffix}")
        self.username = os.getenv(f"CONFLUENCE_USERNAME{suffix}")
        self.api_token = os.getenv(f"CONFLUENCE_API_TOKEN{suffix}")
        self.spaces = os.getenv(f"CONFLUENCE_SPACES{suffix}", "").split(",")


class VectorStoreConfig:
    def __init__(self, profile="internal-confluence"):
        suffix = f"_{profile}" if profile != "internal-confluence" else ""
        self.persist_directory = os.getenv(f"CHROMA_PERSIST_DIRECTORY{suffix}")
        self.embedding_model = os.getenv(f"EMBEDDING_MODEL{suffix}", "text-embedding-3-small")


class OpenAIConfig:
    def __init__(self, profile="internal-confluence"):
        suffix = f"_{profile}" if profile != "internal-confluence" else ""
        self.api_key = os.getenv(f"OPENAI_API_KEY{suffix}", os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv(f"OPENAI_MODEL{suffix}", "gpt-4o-mini")


def get_confluence_client(profile="internal-confluence"):
    config = ConfluenceConfig(profile)
    return Confluence(
        url=config.url,
        username=config.username,
        password=config.api_token,
        cloud=True
    )


def get_embeddings_model(profile="internal-confluence"):
    config = OpenAIConfig(profile)
    vector_config = VectorStoreConfig(profile)
    return OpenAIEmbeddings(
        model=vector_config.embedding_model,
        openai_api_key=config.api_key
    )


def get_llm(profile="internal-confluence"):
    config = OpenAIConfig(profile)
    return ChatOpenAI(
        model=config.model,
        temperature=0,
        openai_api_key=config.api_key
    )


def get_confluence_spaces(profile="internal-confluence"):
    config = ConfluenceConfig(profile)
    return config.spaces


def get_chroma_directory(profile="internal-confluence"):
    config = VectorStoreConfig(profile)
    return config.persist_directory
