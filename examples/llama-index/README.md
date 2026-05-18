# Tools with LlamaIndex functionalities

The following examples leverage tools in the `docling-mcp` server which are leverage LlamaIndex functionalities.

## Requirements

The Docling MCP tools can be executed with any mcp-compatible agents system. For these examples we use:

- A [Llama Stack](https://github.com/llamastack/llama-stack) backend (providing Responses API).
- The [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) library.

## Setup

1. Follow the steps in the [Llama Stack examples](../llama-stack/) for starting the server.

2. Create a virtual environment with [uv](https://docs.astral.sh/uv/) and install the dependencies.

    ```sh
    uv venv
    uv pip install "docling-mcp[llama-index-rag]" openai-agents ipykernel notebook rich
    ```

3. Start the `docling-mcp` server enabling the `conversion` and `llama-index-rag` group:

    ```sh
    docling-mcp-server --transport streamable-http --port 8000 --host 0.0.0.0 conversion llama-index-rag
    ```

## Examples

- [rag_example.ipynb](./rag_example.ipynb)
