<p align="center">
  <a href="https://github.com/docling-project/docling-mcp">
    <img loading="lazy" alt="Docling" src="https://github.com/docling-project/docling-mcp/raw/main/docs/assets/docling_mcp.png" width="40%"/>
  </a>
</p>

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
[![MCP Registry](https://img.shields.io/badge/listed_in-MCP_Registry-green?logo=modelcontextprotocol)](https://registry.modelcontextprotocol.io)

A document processing service using the Docling-MCP library and MCP (Model Context Protocol) for tool integration.


## Overview

[Docling](https://github.com/docling-project/docling) MCP is a service that provides tools for document conversion, processing and generation. It uses the Docling library to convert PDF documents into structured formats and provides a caching mechanism to improve performance. The service exposes functionality through a set of tools that can be called by client applications.

## Compatibility

| docling-mcp | MCP Python SDK |
|---|---|
| `>=3.0.0` | `mcp>=2.0.0` |
| `>=2.0.0,<3.0.0` | `mcp>=1.9.4,<2.0.0` |

If your MCP client application has not yet migrated to MCP SDK v2, pin:

```bash
pip install "docling-mcp<3.0.0"
```

See [MIGRATION.md](MIGRATION.md) for the full migration guide.

## Installation Options

### Remote Mode (Recommended - Lightweight)

For users with access to Docling Serve API:

> **Getting Docling Serve**: Visit [docling-serve](https://github.com/docling-project/docling-serve) for installation guides. You can deploy it from published container images or look for managed Docling SaaS offerings.

```bash
pip install docling-mcp
```

Then configure your environment:
```bash
export DOCLING_MCP_SERVICE_URL=https://your-docling-service.example.com
export DOCLING_MCP_SERVICE_API_KEY=your-api-key-here
export DOCLING_MCP_CONVERSION_MODE=remote
```

### Local Mode (Full Features)

For users who need local conversion or don't have Docling Serve access:

```bash
pip install docling-mcp[local]
```

Then configure your environment:
```bash
export DOCLING_MCP_CONVERSION_MODE=local
```

By default, local mode uses Docling's standard OCR/layout pipeline. To use the
VLM pipeline instead (e.g. [granite-docling](https://huggingface.co/ibm-granite/granite-docling-258M)
served by a local [Ollama](https://ollama.com) instance), set:

```bash
export DOCLING_MCP_USE_VLM=true
export DOCLING_MCP_VLM_HOST=http://localhost:11434
```

### Hybrid Mode (Best of Both)

Install with local support and enable automatic fallback:

```bash
pip install docling-mcp[local]
```

Configure for remote with fallback:
```bash
export DOCLING_MCP_SERVICE_URL=https://your-docling-service.example.com
export DOCLING_MCP_CONVERSION_MODE=remote
export DOCLING_MCP_FALLBACK_TO_LOCAL=true
```

## Features

- Conversion tools:
    - PDF document conversion to structured JSON format ([DoclingDocument][docling_document])
- Generation tools:
    - Document generation in DoclingDocument, which can be exported to multiple formats
- Local document caching for improved performance
- Support for local files and URLs as document sources
- Memory management for handling large documents
- Logging system for debugging and monitoring
- RAG applications with Milvus upload and retrieval

## Configuration

All settings use the `DOCLING_MCP_` prefix and can be supplied as environment
variables, in a `.env` file in the working directory, or via the `env` block of
your MCP client config. Copy [`.env.example`](.env.example) as a starting point.

### Conversion mode

| Variable | Default | Description |
|---|---|---|
| `DOCLING_MCP_CONVERSION_MODE` | `remote` | `remote` or `local` |

### Remote service (required when `DOCLING_MCP_CONVERSION_MODE=remote`)

| Variable | Default | Description |
|---|---|---|
| `DOCLING_MCP_SERVICE_URL` | — | URL of the Docling Serve instance |
| `DOCLING_MCP_SERVICE_API_KEY` | — | API key for the service |
| `DOCLING_MCP_SERVICE_TIMEOUT` | `300.0` | Request timeout in seconds |
| `DOCLING_MCP_SERVICE_MAX_RETRIES` | `3` | Max retry attempts |
| `DOCLING_MCP_FALLBACK_TO_LOCAL` | `false` | Fall back to local if service is unreachable (requires `docling-mcp[local]`) |

### Conversion pipeline (applies to both modes)

| Variable | Default | Description |
|---|---|---|
| `DOCLING_MCP_KEEP_IMAGES` | `false` | Retain page images in output |
| `DOCLING_MCP_IMAGES_SCALE` | `1.0` | Image scale factor (increase to avoid tensor padding errors) |
| `DOCLING_MCP_DO_OCR` | `true` | Run OCR pipeline |
| `DOCLING_MCP_DO_TABLE_STRUCTURE` | `true` | Detect table structure |

### Local VLM pipeline

| Variable | Default | Description |
|---|---|---|
| `DOCLING_MCP_USE_VLM` | `false` | Use the `granite_docling` VLM pipeline through Ollama |
| `DOCLING_MCP_VLM_HOST` | `http://localhost:11434` | Base URL of the Ollama server |

### LlamaIndex RAG (`--tools llama-index-rag`)

| Variable | Default | Description |
|---|---|---|
| `DOCLING_MCP_LI_API_BASE` | `http://127.0.0.1:1234/v1` | OpenAI-compatible LLM endpoint |
| `DOCLING_MCP_LI_API_KEY` | `none` | API key for the LLM endpoint |
| `DOCLING_MCP_LI_MODEL_ID` | `ibm/granite-3.2-8b` | LLM model identifier |
| `DOCLING_MCP_LI_EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | HuggingFace embedding model |

### LlamaStack (`--tools llama-stack-rag` / `--tools llama-stack-ie`)

| Variable | Default | Description |
|---|---|---|
| `DOCLING_MCP_LLS_URL` | `http://localhost:8321` | LlamaStack server URL |
| `DOCLING_MCP_LLS_VDB_EMBEDDING` | `all-MiniLM-L6-v2` | Embedding model for vector DB |
| `DOCLING_MCP_LLS_EXTRACTION_MODEL` | `openai/gpt-oss-20b` | Model used for structured extraction |

### Setting variables in an MCP client config

```json
{
  "mcpServers": {
    "docling": {
      "command": "uvx",
      "args": [
        "--from=docling-mcp",
        "docling-mcp-server"
      ],
      "env": {
        "DOCLING_MCP_CONVERSION_MODE": "remote",
        "DOCLING_MCP_SERVICE_URL": "https://your-docling-service.example.com",
        "DOCLING_MCP_SERVICE_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

## Getting started

The easiest way to install Docling MCP and connect it to your client is by launching it via [uvx](https://docs.astral.sh/uv/).

Depending on the transfer protocol required, specify the argument `--transport`, for example

- **`stdio`** used e.g. in Claude for Desktop and LM Studio 

    ```sh
    uvx --from docling-mcp docling-mcp-server --transport stdio
    ```

- **`sse`** used e.g. in Llama Stack

    ```sh
    uvx --from docling-mcp docling-mcp-server --transport sse
    ```


- **`streamable-http`** used e.g. in containers setup

    ```sh
    uvx --from docling-mcp docling-mcp-server --transport streamable-http
    ```

More options are available, e.g. the selection of which toolgroup to launch. Use the `--help` argument to inspect all the CLI options.

For developing the MCP tools further, please refer to the [Developing](CONTRIBUTING.md#developing) section of CONTRIBUTING.md for instructions.

## Integration with MCP clients

One of the easiest ways to experiment with the tools provided by Docling MCP is to leverage an AI desktop client with MCP support.
Most of these clients use a common config interface. Adding Docling MCP in your favorite client is usually as simple as adding the following entry in the configuration file.

```json
{
  "mcpServers": {
    "docling": {
      "command": "uvx",
      "args": [
        "--from=docling-mcp",
        "docling-mcp-server"
      ]
    }
  }
} 
```

When using **[Claude for Desktop](https://claude.ai/download)**, simply edit the config file `claude_desktop_config.json` with the snippet above or the example provided [here](docs/integrations/claude_desktop_config.json).

In **[LM Studio](https://lmstudio.ai/)**, edit the `mcp.json` file with the appropriate section or simply click on the button below for a direct install.

[![Add MCP Server docling to LM Studio](https://files.lmstudio.ai/deeplink/mcp-install-light.svg)](https://lmstudio.ai/install-mcp?name=docling&config=eyJjb21tYW5kIjoidXZ4IiwiYXJncyI6WyItLWZyb209ZG9jbGluZy1tY3AiLCJkb2NsaW5nLW1jcC1zZXJ2ZXIiXX0%3D)

Other integrations are described in the [integrations] page.

## Examples

### Converting documents

Example of prompt for converting PDF documents:

```prompt
Convert the PDF document at <provide file-path> into DoclingDocument and return its document-key.
```

### Generating documents

Example of prompt for generating new documents:

```prompt
I want you to write a Docling document. To do this, you will create a document first by invoking `create_new_docling_document`. Next you can add a title (by invoking `add_title_to_docling_document`) and then iteratively add new section-headings and paragraphs. If you want to insert lists (or nested lists), you will first open a list (by invoking `open_list_in_docling_document`), next add the list_items (by invoking `add_listitem_to_list_in_docling_document`). After adding list-items, you must close the list (by invoking `close_list_in_docling_document`). Nested lists can be created in the same way, by opening and closing additional lists.

During the writing process, you can check what has been written already by calling the `export_docling_document_to_markdown` tool, which will return the currently written document. At the end of the writing, you must save the document and return me the filepath of the saved document.

The document should investigate the impact of tokenizers on the quality of LLMs.
```

## Contributing

We welcome external contributions. See [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to get started.

## License

The Docling MCP codebase is under MIT license. For individual model usage, please refer to the model licenses found in the original packages.

## LF AI & Data

Docling and Docling MCP is hosted as a project in the [LF AI & Data Foundation](https://lfaidata.foundation/projects/).

**IBM ❤️ Open Source AI**: The project was started by the AI for knowledge team at IBM Research Zurich.

[docling_document]: https://docling-project.github.io/docling/concepts/docling_document/
[integrations]: https://docling-project.github.io/docling-mcp/integrations/

<!-- MCP registry ownership marker (proves this PyPI package maps to the server name). -->
<!-- mcp-name: io.github.docling-project/docling-mcp -->
