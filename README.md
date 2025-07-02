# Docling MCP: making docling agentic 

[![PyPI version](https://img.shields.io/pypi/v/docling-mcp)](https://pypi.org/project/docling-mcp/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/docling-mcp)](https://pypi.org/project/docling-mcp/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pydantic v2](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pydantic/pydantic/main/docs/badge/v2.json)](https://pydantic.dev)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![License MIT](https://img.shields.io/github/license/docling-project/docling-mcp)](https://opensource.org/licenses/MIT)
[![PyPI Downloads](https://static.pepy.tech/badge/docling-mcp/month)](https://pepy.tech/projects/docling-mcp)
[![LF AI & Data](https://img.shields.io/badge/LF%20AI%20%26%20Data-003778?logo=linuxfoundation&logoColor=fff&color=0094ff&labelColor=003778)](https://lfaidata.foundation/projects/)

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
- Milvus upload and retrieval

## Getting started

Install dependencies

```sh
uv sync
```

Install the docling_mcp package

```sh
uv pip install -e .
```

After installing the dependencies (`uv sync`), you can expose the tools of Docling by running

```sh
uv run docling-mcp-server
```

## Integration with Claude for Desktop

One of the easiest ways to experiment with the tools provided by Docling-MCP is to leverage [Claude for Desktop](https://claude.ai/download).
Once installed, extend Claude for Desktop so that it can read from your computer’s file system, by following the [For Claude Desktop Users](https://modelcontextprotocol.io/quickstart/user) tutorial.

To enable Claude for Desktop with Docling MCP, simply edit the config file `claude_desktop_config.json` (located at `~/Library/Application Support/Claude/claude_desktop_config.json` in MacOS) and add a new item in the `mcpServers` key with the details of a Docling MCP server. You can find an example of those details [here](docs/integrations/claude_desktop_config.json).


## Running as streamable-http

Start the server using the following command

```sh
uv run docling-mcp-server --transport streamable-http --http-port 8000
```

## Converting documents

Example of prompt for converting PDF documents:

```prompt
Convert the PDF document at <provide file-path> into DoclingDocument and return its document-key.
```

## Generating documents

Example of prompt for generating new documents:

```prompt
I want you to write a Docling document. To do this, you will create a document first by invoking `create_new_docling_document`. Next you can add a title (by invoking `add_title_to_docling_document`) and then iteratively add new section-headings and paragraphs. If you want to insert lists (or nested lists), you will first open a list (by invoking `open_list_in_docling_document`), next add the list_items (by invoking `add_listitem_to_list_in_docling_document`). After adding list-items, you must close the list (by invoking `close_list_in_docling_document`). Nested lists can be created in the same way, by opening and closing additional lists.

During the writing process, you can check what has been written already by calling the `export_docling_document_to_markdown` tool, which will return the currently written document. At the end of the writing, you must save the document and return me the filepath of the saved document.

The document should investigate the impact of tokenizers on the quality of LLM's.
```

## Applications

### Milvus RAG configuration

Copy the .env.example file to .env in the root of the project.

```sh
cp .env.example .env
```

If you want to use the RAG Milvus functionality edit the new .env file to set both environment variables.

```text
RAG_ENABLED=true
OLLAMA_MODEL=granite3.2:latest
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
```

Note:

ollama can be downloaded here https://ollama.com/. Once you have ollama download the model you want to use and then add the model string to the .env file.

For example we are using `granite3.2:latest` to perform the RAG search.

To download this model run:

```sh
ollama pull granite3.2:latest
```

When using the docling-mcp server with RAG this would be a simple example prompt:

```prompt
Process this file /Users/name/example/mock.pdf 

Upload it to the vector store. 

Then summarize xyz that is contained within the document.
```

Known issues

When restarting the MCP client (e.g. Claude desktop) the client sometimes errors due to the `.milvus_demo.db.lock` file. Delete this before restarting.


## License

The Docling-MCP codebase is under MIT license. For individual model usage, please refer to the model licenses found in the original packages.

## LF AI & Data

Docling and Docling-MCP is hosted as a project in the [LF AI & Data Foundation](https://lfaidata.foundation/projects/).

**IBM ❤️ Open Source AI**: The project was started by the AI for knowledge team at IBM Research Zurich.

[docling_document]: https://docling-project.github.io/docling/concepts/docling_document/
[integrations]: https://docling-project.github.io/docling-mcp/integrations/