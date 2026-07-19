"""Tools for working with the collection of stored documents."""

from dataclasses import dataclass
from typing import Annotated

from mcp.types import ToolAnnotations
from pydantic import Field

from docling_mcp.logger import setup_logger
from docling_mcp.shared import local_document_cache, mcp

# Create a default project logger
logger = setup_logger()


@dataclass
class ConvertedDocumentInfo:
    """Description of one converted document available in the local cache."""

    document_key: Annotated[
        str,
        Field(description="The unique identifier of the document in the local cache."),
    ]
    name: Annotated[
        str | None,
        Field(description="The document name, if available."),
    ]
    source_filename: Annotated[
        str | None,
        Field(description="The filename of the original source."),
    ]
    num_pages: Annotated[
        int,
        Field(description="The number of pages, or 0 when not applicable."),
    ]
    stored_at: Annotated[
        str | None,
        Field(
            description=(
                "UTC timestamp when the document was persisted, or null when "
                "it has not been persisted."
            )
        ),
    ]


@mcp.tool(
    title="List converted documents",
    structured_output=True,
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False),
)
def list_converted_documents() -> list[ConvertedDocumentInfo]:
    """List all converted documents available in the local cache.

    This tool returns one entry per converted document, including documents
    persisted by earlier server sessions. In-progress authored documents are
    not listed. Use the returned document_key with the manipulation and export
    tools instead of re-converting a source.
    """
    return [
        ConvertedDocumentInfo(
            document_key=record.document_key,
            name=record.name,
            source_filename=record.source_filename,
            num_pages=record.num_pages,
            stored_at=record.stored_at,
        )
        for record in local_document_cache.list_metadata()
        if record.source_filename is not None or record.stored_at is not None
    ]
