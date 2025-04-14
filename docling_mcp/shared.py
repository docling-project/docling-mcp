from mcp.server.fastmcp import FastMCP
import os

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.core.node_parser import MarkdownNodeParser
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings

from dotenv import load_dotenv

load_dotenv()

from docling_core.types.doc.document import (
    DoclingDocument,
    NodeItem,
    # DocItem,
    # GroupItem
)

# Create a single shared FastMCP instance
mcp = FastMCP("docling")

# Define your shared cache here if it's used by multiple tools
local_document_cache: dict[str, DoclingDocument] = {}
local_stack_cache: dict[str, list[NodeItem]] = {}

if os.getenv("RAG_ENABLED") == "true" and os.getenv("OLLAMA_MODEL") != "" and os.getenv("EMBEDDING_MODEL") != "":
    Settings.embed_model = HuggingFaceEmbedding(model_name=os.getenv("EMBEDDING_MODEL"))
    Settings.llm = Ollama(model=os.getenv("OLLAMA_MODEL"), request_timeout=120.0)

    node_parser = MarkdownNodeParser()

    db = chromadb.PersistentClient(path="./chroma_db")

    chroma_collection = db.get_or_create_collection("docling-mcp-example")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
