# LLama Stack examples for creating agents using docling-mcp tools

## Requirements

The following applications are required in this example. Refer to their documentation:

- [Ollama](https://github.com/ollama/ollama)
- [Podman](https://podman.io/docs/installation)

## Setup

### Run Llama Stack

As a simple starting point, we will use the [Ollama distribution](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/ollama.html) which allows Llama Stack to easily run locally.
Other distributions (or custom stack builds) will work very similarly. See a complete list in the [Llama Stack docs](https://llama-stack.readthedocs.io/en/latest/distributions/list_of_distributions.html).

1. Pull and load the inference model with Ollama:

  ```shell
  export INFERENCE_MODEL="meta-llama/Llama-3.2-3B-Instruct"
  # ollama names this model differently, and we must use the ollama name when loading the model
  export OLLAMA_INFERENCE_MODEL="llama3.2:3b-instruct-fp16"
  
  ollama run $OLLAMA_INFERENCE_MODEL --keepalive 60m
  ```

2. Ensure that the local folder `~/.llama` exists:
  
  ```shell
  mkdir -p ~/.llama
  ```

3. Run Llama Stack with Ollama as the inference provider, via Podman:

   ```shell
   export LLAMA_STACK_PORT=8321
  
   podman run \
     -it \
     --pull always \
     -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
     -v ~/.llama:/root/.llama \
     llamastack/distribution-ollama \
     --port $LLAMA_STACK_PORT \
     --env INFERENCE_MODEL=$INFERENCE_MODEL \
     --env OLLAMA_URL=http://host.containers.internal:11434
  ```

### Connect to Docling MCP server

1. Clone this repository [docling-mcp](https://github.com/docling-project/docling-mcp)

   ```shell
   uv sync --extra llama-stack
   ```

2. Make sure the Docling MCP server is running with the `sse` option (default)

   ```shell
   uv run docling-mcp-server --transport sse --http-port 8000
   ```

3. Register the Docling tools

   You can use the Tool Group Management of the [llama (client-side) CLI](https://llama-stack.readthedocs.io/en/latest/references/llama_stack_client_cli_reference.html#)
   to register the Docling MCP server tools

   ```shell
   uvx --from llama-stack-client llama-stack-client toolgroups register "mcp::docling" \
     --provider-id="model-context-protocol" \
     --mcp-endpoint="http://host.containers.internal:8000/sse"
   ```

   Alternatively, you can run this script from a python session started with `uvx --with llama-stack python`.

   ```py
   from llama_stack_client import LlamaStackClient
   client = LlamaStackClient(base_url="http://localhost:8321")
   client.toolgroups.register(
     toolgroup_id="mcp::docling",
     provider_id="model-context-protocol",
     mcp_endpoint={"uri": "http://host.containers.internal:8000/sse"},
   )
   exit()
   ```

4. Inspect the tools

   ```shell
   uvx --with llama-stack --from llama-stack-client llama-stack-client toolgroups list
   ```

   ```console

    ┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    ┃ identifier             ┃ provider_id            ┃ args ┃ mcp_endpoint                                                ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
    │ builtin::rag           │ rag-runtime            │ None │ None                                                        │
    │ builtin::websearch     │ tavily-search          │ None │ None                                                        │
    │ builtin::wolfram_alpha │ wolfram-alpha          │ None │ None                                                        │
    │ mcp::docling           │ model-context-protocol │ None │ McpEndpoint(uri='http://host.containers.internal:8000/see') │
    └────────────────────────┴────────────────────────┴──────┴─────────────────────────────────────────────────────────────┘
   
   ```

## Use the Llama Stack agents

### Playground UI

Llama Stack provides a demonstration playground UI ([Llama Stack Playground](https://llama-stack.readthedocs.io/en/latest/playground/)). At the moment the UI is not distributed and has to be built from sources.

The example [playground-ui](./playground-ui/) provides the simple instructions to get it working locally.

1. Build and run the Llama Stack Playground by followign the instructions on [playground-ui](./playground-ui/)
2. Access the UI on http://localhost:8501/tools
3. The **docling** MCP server will show up on the **Available ToolGroups** section of the UI.

### Test the agent programmatically

The same results are achieved when calling the Llama Stack agents runtime from a script. Below are a few example notebooks to get started.

- [TBA](./)
