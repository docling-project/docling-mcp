"""This module defines the RAG tools using Llama Stack."""

from dataclasses import dataclass
from typing import Annotated

from llama_stack_client.types.vector_io_insert_params import Chunk
from pydantic import Field
from transformers import AutoTokenizer

from docling_core.transforms.chunker.hierarchical_chunker import DocMeta
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

from docling_mcp.logger import setup_logger
from docling_mcp.settings.llama_stack import settings
from docling_mcp.shared import local_document_cache, mcp
from docling_mcp.tools.llama_stack._shared import get_llama_stack_client

logger = setup_logger()


@dataclass
class InsertDocumentOutput:
    """Output of the Docling document content insert tool."""

    vector_db_id: Annotated[
        str,
        Field(description="The vectordb identifier where the data has been inserted."),
    ]


@mcp.tool()
def insert_document_to_vectordb(
    document_key: Annotated[
        str,
        Field(description="The unique identifier of the document in the local cache."),
    ],
    vector_db_id: Annotated[str, Field(description="The target vectordb identifier.")],
) -> InsertDocumentOutput:
    """Insert the document in the vecrtordb.

    This tool chunks and ingests a Docling document that exists in the local cache into
    a vectordb. The vectordb can then be used for knowledge searches.
    """
    logger.debug(f"{document_key=}")
    logger.debug(f"{vector_db_id=}")

    if document_key not in local_document_cache:
        doc_keys = ", ".join(local_document_cache.keys())
        raise ValueError(
            f"document-key: {document_key} is not found. Existing document-keys are: {doc_keys}"
        )

    doc = local_document_cache[document_key]
    doc_id = str(doc.origin.binary_hash) if doc.origin is not None else document_key

    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(  # type: ignore
            pretrained_model_name_or_path=f"sentence-transformers/{settings.vdb_embedding}"
        ),
    )
    chunker = HybridChunker(tokenizer=tokenizer)

    chunk_iter = chunker.chunk(dl_doc=doc)

    ls_chunks: list[Chunk] = []
    for chunk in chunk_iter:
        meta = DocMeta.model_validate(chunk.meta)

        enriched_text = chunker.contextualize(chunk=chunk)

        token_count = tokenizer.count_tokens(enriched_text)
        chunk_dict: Chunk = {
            "content": enriched_text,
            "metadata": {
                "document_id": doc_id,
                "token_count": token_count,
                "doc_items": [item.self_ref for item in meta.doc_items],
            },
        }
        ls_chunks.append(chunk_dict)

    client = get_llama_stack_client()
    client.vector_io.insert(
        vector_db_id=vector_db_id,
        chunks=ls_chunks,
    )

    return InsertDocumentOutput(vector_db_id=vector_db_id)
