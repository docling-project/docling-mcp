"""This module defines applications."""

import os
from typing import Any

from docling_core.types.doc.document import DoclingDocument

from docling_mcp.logger import setup_logger
from docling_mcp.shared import local_document_cache, mcp

logger = setup_logger()

if (
    os.getenv("RAG_ENABLED") == "true"
    and os.getenv("OLLAMA_MODEL") != ""
    and os.getenv("EMBEDDING_MODEL") != ""
):
    import json

    from llama_index.core import Document, StorageContext, VectorStoreIndex
    from llama_index.core.base.response.schema import (
        RESPONSE_TYPE,
        Response,
    )
    from mcp.shared.exceptions import McpError
    from mcp.types import INTERNAL_ERROR, ErrorData

    from docling_mcp.shared import local_index_cache, milvus_vector_store, node_parser

    @mcp.tool()
    def export_docling_document_to_vector_db(document_key: str) -> str:
        """Exports a document from the local document cache to a vector database for search capabilities.

        This tool converts a Docling document that exists in the local cache into markdown format,
        then loads it into a vector database index. This allows the document to be searched using
        semantic search techniques.

        Args:
            document_key (str): The unique identifier for the document in the local cache.

        Returns:
            str: A confirmation message indicating the document was successfully indexed.

        Raises:
            ValueError: If the specified document_key does not exist in the local cache.

        Example:
            export_docling_document_to_vector_db("doc123")
        """
        if document_key not in local_document_cache:
            doc_keys = ", ".join(local_document_cache.keys())
            raise ValueError(
                f"document-key: {document_key} is not found. Existing document-keys are: {doc_keys}"
            )

        docling_document: DoclingDocument = local_document_cache[document_key]
        document_dict: dict[str, Any] = docling_document.export_to_dict()
        document_json: str = json.dumps(document_dict)

        document = Document(
            text=document_json,
            metadata={"filename": docling_document.name},
        )

        index = VectorStoreIndex.from_documents(
            documents=[document],
            transformations=[node_parser],
            storage_context=StorageContext.from_defaults(
                vector_store=milvus_vector_store
            ),
        )

        index.insert(document)

        local_index_cache["milvus_index"] = index

        return f"Successful initialisation for document with id {document_key}"

    @mcp.tool()
    def search_documents(query: str) -> str:
        """Searches through previously uploaded and indexed documents using semantic search.

        This function retrieves relevant information from documents that have been processed
        and added to the vector database. It uses semantic similarity to find content that
        best matches the query, rather than simple keyword matching.

        Args:
            query (str): The search query text used to find relevant information in the indexed documents.

        Returns:
            str: A string containing the relevant contextual information retrieved from the documents
                 that best matches the query.

        Example:
            search_documents("What are the main findings about climate change?")
        """
        index = local_index_cache["milvus_index"]

        query_engine = index.as_query_engine()
        response: RESPONSE_TYPE = query_engine.query(query)

        if isinstance(response, Response):
            if response.response is not None:
                return response.response
            else:
                raise McpError(
                    ErrorData(
                        code=INTERNAL_ERROR,
                        message="Response object has no response content",
                    )
                )
        else:
            raise McpError(
                ErrorData(
                    code=INTERNAL_ERROR,
                    message=f"Unexpected response type: {type(response)}",
                )
            )
