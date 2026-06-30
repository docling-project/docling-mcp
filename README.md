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

A document processing service using the Docling-MCP library and MCP (Model Context Protocol) for tool integration.


## Overview

[Docling](https://github.com/docling-project/docling) MCP is a service that provides tools for document conversion, processing and generation. It uses the Docling library to convert PDF documents into structured formats and provides a caching mechanism to improve performance. The service exposes functionality through a set of tools that can be called by client applications.

## 🆕 What's New in v2.0

**Major Architecture Update**: Docling MCP v2.0 introduces a hybrid architecture with support for both remote API and local conversion modes:

- **🚀 90% Size Reduction**: Base package is now ~50MB (down from ~500MB)
- **⚡ Faster Installation**: No model downloads required for default remote mode
- **🌐 Remote API Support**: Use Docling Serve for scalable cloud-based conversion
- **💻 Local Mode Available**: Install `[local]` extra for offline/local conversion
- **🔄 Automatic Fallback**: Optional fallback from remote to local mode
- **🎯 Flexible Configuration**: Choose the mode that fits your needs

**Migration**: Upgrading from v1.x? See [MIGRATION_v2.md](MIGRATION_v2.md) for detailed instructions.

## Installation Options

### Remote Mode (Recommended - Lightweight)

For users with access to Docling Serve API:

> **Getting Docling Serve**: Visit [docling-serve](https://github.com/docling-project/docling-serve) for installation guides. You can deploy it from published container images or look for managed Docling SaaS offerings.

```bash
pip install docling-mcp
```

Then configure your environment:
```bash
export DOCLING_SERVICE_URL=https://your-docling-service.example.com
export DOCLING_SERVICE_API_KEY=your-api-key-here
export DOCLING_CONVERSION_MODE=remote
```

### Local Mode (Full Features)

For users who need local conversion or don't have Docling Serve access:

```bash
pip install docling-mcp[local]
```

Then configure your environment:
```bash
export DOCLING_CONVERSION_MODE=local
```

### Hybrid Mode (Best of Both)

Install with local support and enable automatic fallback:

```bash
pip install docling-mcp[local]
```

Configure for remote with fallback:
```bash
export DOCLING_SERVICE_URL=https://your-docling-service.example.com
export DOCLING_CONVERSION_MODE=remote
export DOCLING_FALLBACK_TO_LOCAL=true
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

Docling MCP can be configured using environment variables. The following options are available:

- **`DOCLING_MCP_KEEP_IMAGES`**: Set to `true` to keep page images in the converted documents (default: `false`)
- **`DOCLING_MCP_IMAGES_SCALE`**: Scale factor for image processing to avoid tensor padding errors (default: `1.0`). Adjusting this value (e.g., `1.0`, `2.0`) can help prevent batching issues when processing images or PDFs.

To set these variables, you can:
1. Create a `.env` file in your working directory
2. Set them as environment variables in your system
3. Pass them in the MCP client configuration (see examples below)

Example `.env` file:
```
DOCLING_MCP_KEEP_IMAGES=true
DOCLING_MCP_IMAGES_SCALE=2.0
```

Example MCP client configuration with environment variables:
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
        "DOCLING_MCP_IMAGES_SCALE": "2.0"
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

## Pipeline overrides (per-call + env-var fallback)

The convert tool exposes four optional pipeline overrides. When omitted, each
falls back to its environment-variable default — but for one-off calls (born-
digital PDF that needs no OCR, single document where you also want page
images, switching between local and remote mid-session), passing them per-call
avoids server-restart churn.

```python
docling-pr-95:convert_document_into_docling_document(
    source="/path/to/born-digital.pdf",
    do_ocr=False,              # skip OCR for clean PDFs (huge speed-up)
    do_table_structure=False,  # skip table layout analysis
    keep_images=True,          # enable page_thumbnail downstream
    conversion_mode="local",   # use local converter even if env says remote
)
```

| Param | Env-var default | Behavior |
|---|---|---|
| `do_ocr` | `DOCLING_MCP_DO_OCR` (default `true`) | Run OCR over every page. Disable for born-digital PDFs to skip a multi-minute step. |
| `do_table_structure` | `DOCLING_MCP_DO_TABLE_STRUCTURE` (default `true`) | Detect table layout. Disable for plain-text PDFs. |
| `keep_images` | `DOCLING_MCP_KEEP_IMAGES` (default `false`) | Generate per-page images at convert time. Required if you'll call `page_thumbnail` afterward — images are not retrievable after the fact. |
| `conversion_mode` | `DOCLING_CONVERSION_MODE` (default `remote`) | `local` uses `DocumentConverter` from the `[local]` extra. `remote` requires `DOCLING_SERVICE_URL` to be set. |

Per-call overrides take precedence; env vars are the fallback default for
calls that omit them. The `LocalDocumentConverter` caches one underlying
`DocumentConverter` instance per unique override tuple, so repeated calls
with the same options don't pay the model-load cost twice.

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
