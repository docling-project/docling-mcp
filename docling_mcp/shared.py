import os

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.vector_stores.milvus import MilvusVectorStore
from mcp.server.fastmcp import FastMCP

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

if (
    os.getenv("RAG_ENABLED") == "true"
    and os.getenv("OLLAMA_MODEL") != ""
    and os.getenv("EMBEDDING_MODEL") != ""
):
    embed_model = HuggingFaceEmbedding(model_name=os.getenv("EMBEDDING_MODEL"))
    Settings.embed_model = embed_model
    Settings.llm = Ollama(model=os.getenv("OLLAMA_MODEL"), request_timeout=120.0)

    node_parser = DoclingNodeParser()

    embed_dim = len(embed_model.get_text_embedding("hi"))

    milvus_vector_store = MilvusVectorStore(
        uri="./milvus_demo.db", dim=embed_dim, overwrite=True
    )

    local_index_cache: dict[str, VectorStoreIndex] = {}
