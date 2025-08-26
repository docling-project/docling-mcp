"""Tools performing structured information extraction using Llama Stack model inference."""

import json
from dataclasses import dataclass
from typing import Annotated

from pydantic import Field

from docling_mcp.logger import setup_logger
from docling_mcp.settings.llama_stack import settings
from docling_mcp.shared import local_document_cache, mcp
from docling_mcp.tools.llama_stack._shared import get_llama_stack_client

logger = setup_logger()


@dataclass
class ExtractedContent:
    """Output of the requested key value pairs."""

    result: Annotated[
        dict[str, str],
        Field(
            description="Dictionary of the requested keys with the respective value extracted from the document."
        ),
    ]


@mcp.tool(title="Information extraction from a document")
def information_extraction(
    document_key: Annotated[
        str,
        Field(description="The unique identifier of the document in the local cache."),
    ],
    keys: Annotated[list[str], Field(description="List of keys to be extracted.")],
    descriptions: Annotated[
        list[str],
        Field(
            description="Description of the keys. Must be the same length of keys or it will be ignored."
        ),
    ],
    max_window_size: Annotated[
        int | None,
        Field(
            description="The maximum document size (in characters) to use for the extraction."
        ),
    ] = None,
) -> ExtractedContent:
    """Information extraction from a document.

    This tool searches for the requested keys matching the descriptions in the document.
    """
    logger.debug(f"{document_key=}")
    logger.debug(f"{keys=}")
    logger.debug(f"{descriptions=}")

    if document_key not in local_document_cache:
        doc_keys = ", ".join(local_document_cache.keys())
        raise ValueError(
            f"document-key: {document_key} is not found. Existing document-keys are: {doc_keys}"
        )

    doc = local_document_cache[document_key]
    content_md = doc.export_to_markdown()
    if max_window_size is not None:
        content_md = content_md[:max_window_size]

    prompt = "Extract the following information from the document:\n"
    schema: dict[str, dict[str, str]] = {}
    for key, desc in zip(keys, descriptions, strict=False):
        prompt += f"- {key}: {desc}\n"
        schema[key] = {
            "type": "string",
            "description": desc,
        }
    extraction_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "extraction",
            "schema": {
                "type": "object",
                "properties": schema,
                "required": list(keys),
            },
        },
    }
    prompt = prompt.strip()

    client = get_llama_stack_client()
    chat_completion = client.chat.completions.create(
        model=settings.extraction_model,
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": content_md,
            },
        ],
        response_format=extraction_schema,
    )  # type: ignore[call-overload, unused-ignore]

    response = json.loads(chat_completion.choices[0].message.content)

    return ExtractedContent(result=response)
