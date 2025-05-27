# LLama Stack examples for creating agents using docling-mcp tools

## Setup

As a simple starting point, we will use the [Ollama distribution](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/ollama.html) which allows Llama Stack to easily run locally.
Other distributions (or custom stack builds) will work very similarly. See a complete list in the [Llama Stack docs](https://llama-stack.readthedocs.io/en/latest/distributions/list_of_distributions.html).

Launch Llama Stack:

```shell
export INFERENCE_MODEL="meta-llama/Llama-3.2-3B-Instruct"

# ollama names this model differently, and we must use the ollama name when loading the model
export OLLAMA_INFERENCE_MODEL="llama3.2:3b-instruct-fp16"
ollama run $OLLAMA_INFERENCE_MODEL --keepalive 60m

# launch llama stack
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

### Connect your agents

1. Make sure the Docling MCP server is running with the `sse` option

    ```shell
    docling-mcp-server --transport sse --sse-port 8000
    ```

2. Register the agent (NOT YET WORKING)

    The `llama-stack-client` is current broken and fails the registration of tools. We have to use a Python script, see below.

    ```shell
    uvx --from llama-stack-client llama-stack-client toolgroups register "mcp::docling" \
      --provider-id="model-context-protocol" \
      --mcp-endpoint="http://host.containers.internal:8000/sse"
    ```

2. Register the agent (WORKAROUND)

    As a workaround, we have to run the following script to register the tools.
    This can be execution, e.g. from a python session started with `uvx --from llama-stack-client python`.

    ```py
    from llama_stack_client import LlamaStackClient
    client = LlamaStackClient(base_url="http://localhost:8321")
    client.toolgroups.register(
        toolgroup_id="mcp::docling",
        provider_id="model-context-protocol",
        mcp_endpoint={"uri": "http://host.containers.internal:8000/sse"},
    )
    ```

3. Inspect the tools

    ```shell
    uvx --from llama-stack-client llama-stack-client toolgroups list

    ┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    ┃ identifier             ┃ provider_id            ┃ args ┃ mcp_endpoint                                         ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
    │ builtin::rag           │ rag-runtime            │ None │ None                                                 │
    │ builtin::websearch     │ tavily-search          │ None │ None                                                 │
    │ builtin::wolfram_alpha │ wolfram-alpha          │ None │ None                                                 │
    │ mcp::docling           │ model-context-protocol │ None │ McpEndpoint(uri='http://host.containers.internal:80… │
    └────────────────────────┴────────────────────────┴──────┴──────────────────────────────────────────────────────┘
    ```

## Use the Llama Stack agents

### Playground UI

Llama Stack provides a demonstration playground UI. At the moment the UI is not distributed and has to be built from sources.

The example [playground-ui](./playground-ui/) provides the simple instructions to get it working locally.

1. Build and run the [playground-ui](./playground-ui/)
2. Your agent will show up in the Tools section of the UI.

### Test the agent programmatically

The same results are achieved when calling the Llama Stack agents runtime from a script. Below are a few example notebooks to get started.

- [TBA](./)
