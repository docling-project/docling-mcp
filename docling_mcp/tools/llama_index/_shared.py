"""This module defines the shared components for the Llama Index tools tools."""

from llama_index.core import Settings as LISettings
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.vector_stores.milvus import MilvusVectorStore

from docling_mcp.settings.llama_index import settings

embed_model = HuggingFaceEmbedding(model_name=settings.embedding_model)
LISettings.embed_model = embed_model
LISettings.llm = OpenAILike(
    api_base=settings.api_base,
    api_key=settings.api_key,
    model=settings.model_id,
    request_timeout=120.0,
)

node_parser = DoclingNodeParser()

embed_dim = len(embed_model.get_text_embedding("hi"))

milvus_vector_store = MilvusVectorStore(
    uri="./milvus_demo.db", dim=embed_dim, overwrite=True
)

local_index_cache: dict[str, VectorStoreIndex] = {}
