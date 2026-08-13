"""Test that registered prompts have correct metadata."""

import pytest

from tests.conftest import MCPClient

# Expected prompts: name -> (title, single-line description, argument names)
_EXPECTED_PROMPTS = {
    "generate_docling_document_from_pdf": (
        "Convert document",
        "Convert a local file into a Docling document and return its document key.",
        ["file_path"],
    ),
    "convert_and_summarize": (
        "Convert and summarize",
        "Convert a document and produce a structured summary.",
        ["source"],
    ),
    "convert_directory_and_list": (
        "Convert directory",
        "Convert all files in a directory and list the results.",
        ["directory"],
    ),
    "author_structured_document": (
        "Author structured document",
        "Create a new structured Docling document on a given topic.",
        ["topic", "sections"],
    ),
    "convert_and_rewrite": (
        "Convert and rewrite",
        "Convert a document and rewrite its content following specific instructions.",
        ["source", "instructions"],
    ),
    "review_and_edit_document": (
        "Review and edit",
        "Review the structure of a cached document and interactively edit it.",
        ["document_key"],
    ),
    "find_and_replace_in_document": (
        "Find and replace",
        "Find text in a cached document and replace it at the correct anchor.",
        ["document_key", "search_text", "replacement_text"],
    ),
}


@pytest.mark.anyio
async def test_prompt_metadata(mcp_client: MCPClient) -> None:
    """Prompt descriptions are single-line and arguments carry their own descriptions."""
    prompts = await mcp_client.list_prompts()
    by_name = {p.name: p for p in prompts}

    assert set(by_name) == set(_EXPECTED_PROMPTS), (
        f"Unexpected prompt names: {set(by_name) - set(_EXPECTED_PROMPTS)}"
    )

    for name, (title, description, arg_names) in _EXPECTED_PROMPTS.items():
        prompt = by_name[name]

        # Description must be exactly the summary line — no newlines, no "Args:" section.
        assert "\n" not in (prompt.description or ""), (
            f"Prompt '{name}' description contains newlines: {prompt.description!r}"
        )
        assert (prompt.description or "").strip() == description, (
            f"Prompt '{name}' description mismatch"
        )

        # Title is set explicitly via @mcp.prompt(title=...).
        assert prompt.title == title, f"Prompt '{name}' title mismatch"

        # Every argument must carry a non-empty description (from Field(description=...)).
        actual_arg_names = [a.name for a in (prompt.arguments or [])]
        assert actual_arg_names == arg_names, (
            f"Prompt '{name}' argument names mismatch: {actual_arg_names} != {arg_names}"
        )
        for arg in prompt.arguments or []:
            assert arg.description, (
                f"Prompt '{name}', argument '{arg.name}' has no description"
            )
