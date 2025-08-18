"""This module defines the RAG with milvus tools."""

import json
from dataclasses import dataclass
from typing import Annotated, Any

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.base.response.schema import (
    RESPONSE_TYPE,
    Response,
)
from mcp.shared.exceptions import McpError
from mcp.types import INTERNAL_ERROR, ErrorData
from pydantic import Field

from docling_core.types.doc.document import DoclingDocument

from docling_mcp.logger import setup_logger
from docling_mcp.shared import (
    local_document_cache,
    mcp,
)
from docling_mcp.tools.llama_index._shared import (
    local_index_cache,
    milvus_vector_store,
    node_parser,
)

logger = setup_logger()


@mcp.tool(title="Export Docling document to vector database")
def export_docling_document_to_vector_db(
    document_key: Annotated[
        str,
        Field(description="The unique identifier of the document in the local cache."),
    ],
) -> str:
    """Exports a document from the local document cache to a vector database for search capabilities.

    This tool converts a Docling document that exists in the local cache into markdown format,
    then loads it into a vector database index. This allows the document to be searched using
    semantic search techniques.

    Raises:
        ValueError: If the specified document_key does not exist in the local cache.
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
        storage_context=StorageContext.from_defaults(vector_store=milvus_vector_store),
    )

    index.insert(document)

    local_index_cache["milvus_index"] = index

    return f"Successful initialisation for document with id {document_key}"


@dataclass
class SearchDocumentOutput:
    """Output of the search documents tool."""

    answer: Annotated[
        str,
        Field(
            description="A string containing the relevant contextual information retrieved from the documents that best matches the query."
        ),
    ]

    # TODO: future updates could provide the grounding elements metadata


@mcp.tool(title="Search query in documents")
def search_documents(
    query: Annotated[
        str,
        Field(
            description="The search query text used to find relevant information in the indexed documents."
        ),
    ],
) -> SearchDocumentOutput:
    """Searches through previously uploaded and indexed documents using semantic search.

    This function retrieves relevant information from documents that have been processed
    and added to the vector database. It uses semantic similarity to find content that
    best matches the query, rather than simple keyword matching.
    """
    index = local_index_cache["milvus_index"]

    query_engine = index.as_query_engine()
    response: RESPONSE_TYPE = query_engine.query(query)

    if isinstance(response, Response):
        if response.response is not None:
            return SearchDocumentOutput(answer=response.response)
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
