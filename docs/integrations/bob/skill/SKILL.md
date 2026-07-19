---
name: docling
description: Convert PDF, DOCX, PPTX, XLSX, HTML, and scanned or image-based documents into structured DoclingDocuments with the Docling MCP tools before working with their content. Activate whenever the user references a document file or asks to read, summarize, compare, extract from, or edit one.
---

# Docling-first document handling

The `docling` MCP server converts documents into DoclingDocument, a structured format held in the server's cache and addressed by a `document_key`. Work from that structured output. Do not read document files directly or paste raw document text into the conversation.

This skill requires the `docling` MCP server from the workspace or global MCP configuration. If its tools are not available, stop and tell the user to set it up first, following https://github.com/docling-project/docling-mcp/blob/main/docs/integrations/bob.md.

## When to use this workflow

Use it whenever a task involves a PDF, DOCX, PPTX, XLSX, or HTML file, a scanned document, or an image of a page. Typical signals: the user names a file with one of those extensions, points at a directory of documents, or asks about the content of a report, paper, spec, or contract.

## Workflow

1. Convert first. Call `convert_document_into_docling_document` with the file's absolute path or URL. For a directory of documents, call `convert_directory_files_into_docling_document` once instead of converting file by file.
2. Record the returned `document_key`. A result with `"from_cache": true` means the document was already converted; reuse the key.
3. Inspect structure before reading content. `get_overview_of_document_anchors` returns the document outline with an anchor per item. Use it to locate the relevant sections.
4. Read selectively. `get_text_of_document_item_at_anchor` returns the text of a single item, and `search_for_text_in_document_anchors` finds anchors matching a search string. Only call `export_docling_document_to_markdown` when the task genuinely needs the full text.
5. When asked to change content, edit through anchors with `update_text_of_document_item_at_anchor` or `delete_document_items_at_anchors`, then persist with `save_docling_document`.

## Rules

- Refer to converted documents by `document_key` in every follow-up tool call.
- Quote only the document items needed to answer, and name the anchor they came from. Never paste large verbatim excerpts into the conversation.
- Use absolute file paths. Relative paths resolve against the MCP server process, not the Bob workspace.
- The document cache lives in server memory. If a `document_key` stops resolving after the server restarts, convert the source again.
- Conversion of large or scanned files can take minutes. If a conversion call times out, ask the user to raise Bob's MCP network timeout instead of retrying in a loop.
