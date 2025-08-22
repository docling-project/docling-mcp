# Agents using the Pydantic AI SDK and Docling MCP

The following examples leverage tools in the `docling-mcp` server via the [Pydantic AI](https://ai.pydantic.dev) framework.

## Requirements

The Docling MCP tools can be executed with any mcp-compatible agents system. For these examples we use:

- [Pydantic AI](https://ai.pydantic.dev)

## Setup

Start the `docling-mcp` server (using the default `conversion` and `generation` groups):

```sh
docling-mcp-server --transport streamable-http --port 8000 --host 0.0.0.0
```

## Examples

- [structured_data.ipynb](./structured_data.ipynb) using LM Studio directly
- [structured_data_responses.ipynb](./structured_data.ipynb) using an OpenAI-compatible Responses API


