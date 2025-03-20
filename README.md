# Docling MCP: making docling agentic 

[![PyPI version](https://img.shields.io/pypi/v/docling-mcp)](https://pypi.org/project/docling-mcp/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/docling-mcp)](https://pypi.org/project/docling-mcp/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Pydantic v2](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pydantic/pydantic/main/docs/badge/v2.json)](https://pydantic.dev)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![License MIT](https://img.shields.io/github/license/docling-project/docling-mcp)](https://opensource.org/licenses/MIT)
[![PyPI Downloads](https://static.pepy.tech/badge/docling-mcp/month)](https://pepy.tech/projects/docling-mcp)

A document processing service using the Docling-MCP library and MCP (Message Control Protocol) for tool integration.

 > [!NOTE]
> This is an unstable draft implementation which will quickly evolve.

## Overview

Docling MCP is a service that provides tools for document conversion, processing and generation. It uses the Docling library to convert PDF documents into structured formats and provides a caching mechanism to improve performance. The service exposes functionality through a set of tools that can be called by client applications.

## Features

- conversion tools:
    - PDF document conversion to structured JSON format (DoclingDocument)
- generation tools:
    - Document generation in DoclingDocument, which can be exported to multiple formats
- Local document caching for improved performance
- Support for local files and URLs as document sources
- Memory management for handling large documents
- Logging system for debugging and monitoring

## Getting started

After installing the dependencies (`uv sync`), you can expose the tools of Docling by running,

```sh
uv run python -m docling_mcp.server
```

### Integration into Claude Desktop

One of the easiest ways to experiment with the tools provided by Docling-MCP is to leverage the Claude Desktop. For that. simply update your Claude Desktop config file (located at `~/Library/Application Support/Claude/claude_desktop_config.json`) and add an item (see [here](docs/integrations/claude_desktop_config.json)) to the `mcpServers` key. 

## Converting documents

```prompt
Convert the PDF document at <provide file-path> into DoclingDocument and return me its document-key.
```

## Generating documents

Example prompt for generation:

```prompt
I want you to write a Docling document. To do this, you will create a document first by invoking `create_new_docling_document`. Next you can add a title (by invoking `add_title_to_docling_document`) and then iteratively add new section-headings and paragraphs. If you want to insert lists (or nested lists), you will first open a list (by invoking `open_list_in_docling_document`), next add the list_items (by invoking `add_listitem_to_list_in_docling_document`). After adding list-items, you must close the list (by invoking `close_list_in_docling_document`). Nested lists can be created in the same way, by opening and closing additional lists.

During the writing process, you can check what has been written already by calling the `export_docling_document_to_markdown` tool, which will return the currently written document. At the end of the writing, you must save the document and return me the filepath of the saved document.

The document should investigate the impact of tokenizers on the quality of LLM's.
```

## License

The Docling-MCP codebase is under MIT license. For individual model usage, please refer to the model licenses found in the original packages.

## LF AI & Data

Docling and Docling-MCP is hosted as a project in the [LF AI & Data Foundation](https://lfaidata.foundation/projects/).

**IBM ❤️ Open Source AI**: The project was started by the AI for knowledge team at IBM Research Zurich.

[docling_document]: https://docling-project.github.io/docling/concepts/docling_document/
[integrations]: https://docling-project.github.io/docling-mcp/integrations/