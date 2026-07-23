"""Tools for working with the collection of stored documents."""

from dataclasses import dataclass
from typing import Annotated

from mcp.types import ToolAnnotations
from pydantic import Field

from docling_mcp.logger import setup_logger
from docling_mcp.shared import local_document_cache, mcp
from docling_mcp.store.local import is_valid_document_key

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


@dataclass
class CorpusSearchResult:
    """One ranked match from a search across the converted documents."""

    document_key: Annotated[
        str,
        Field(description="The unique identifier of the document in the local cache."),
    ]
    anchor: Annotated[
        str,
        Field(
            description=(
                "The anchor reference of the matching document item, usable "
                "with the manipulation tools."
            ),
            examples=["#/texts/2"],
        ),
    ]
    snippet: Annotated[
        str,
        Field(description="A short text excerpt around the match."),
    ]
    page: Annotated[
        int | None,
        Field(
            description=(
                "The page number the matching item appears on, or null when "
                "the document has no page provenance."
            )
        ),
    ]


@mcp.tool(
    title="Search across documents",
    structured_output=True,
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False),
)
def search_across_documents(
    query: Annotated[
        str,
        Field(
            description=(
                "The text to search for. Matching is case-insensitive and "
                "word-based; every word in the query must appear in a "
                "matching item."
            )
        ),
    ],
    max_results: Annotated[
        int,
        Field(
            ge=1,
            le=100,
            description="The maximum number of hits to return.",
        ),
    ] = 10,
    document_keys: Annotated[
        list[str] | None,
        Field(
            description=(
                "Restrict the search to these document keys, as returned by "
                "list_converted_documents. Searches all converted documents "
                "when omitted."
            )
        ),
    ] = None,
) -> list[CorpusSearchResult]:
    """Search for text across all converted documents in the local cache.

    This tool returns ranked hits, each carrying the document key, the anchor
    of the matching item, a snippet of the matching text, and the page number
    when the document has page provenance. Pass the anchor to
    get_text_of_document_item_at_anchor to read the full item, and cite the
    document and page from the hit. The search covers converted documents,
    including documents persisted by earlier server sessions; in-progress
    authored documents are not searched.
    """
    if not query.strip():
        raise ValueError("The search query must not be empty.")
    if document_keys is not None:
        for key in document_keys:
            if not is_valid_document_key(key):
                raise ValueError(f"Invalid document key: {key!r}")

    hits = local_document_cache.search_corpus(
        query, max_results=max_results, document_keys=document_keys
    )
    return [
        CorpusSearchResult(
            document_key=hit.document_key,
            anchor=hit.anchor,
            snippet=hit.snippet,
            page=hit.page,
        )
        for hit in hits
    ]
