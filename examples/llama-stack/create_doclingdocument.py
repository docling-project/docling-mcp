import json
import logging

import httpx
from pathlib import Path

from rich.table import Table

from urllib.parse import urlparse

from llama_stack_client import Agent, AgentEventLogger, LlamaStackClient
from llama_stack_client.lib import get_oauth_token_for_mcp_server

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)


def check_model_exists(client: LlamaStackClient, model_id: str) -> bool:
    models = [m for m in client.models.list() if m.model_type == "llm"]
    if model_id not in [m.identifier for m in models]:
        _log.error(f"Model {model_id} not found[/red]")
        _log.error("Available models:[/yellow]")
        for model in models:
            _log.error(f"  - {model.identifier}")
        return False
    return True


def get_and_cache_mcp_headers(
    servers: list[str], cache_file: Path = Path("./cache")
) -> dict[str, dict[str, str]]:
    mcp_headers = {}

    _log.info(f"Using cache file: {cache_file} for MCP tokens")
    tokens = {}
    if cache_file.exists():
        with open(cache_file, "r") as f:
            tokens = json.load(f)
            for server, token in tokens.items():
                mcp_headers[server] = {
                    "Authorization": f"Bearer {token}",
                }

    for server in servers:
        with httpx.Client() as http_client:
            headers = mcp_headers.get(server, {})
            try:
                response = http_client.get(server, headers=headers, timeout=1.0)
            except httpx.TimeoutException:
                # timeout means success since we did not get an immediate 40X
                continue

            if response.status_code in (401, 403):
                _log.info(f"Server {server} requires authentication, getting token")
                token = get_oauth_token_for_mcp_server(server)
                if not token:
                    _log.error(f"No token obtained for {server}")
                    return

                tokens[server] = token
                mcp_headers[server] = {
                    "Authorization": f"Bearer {token}",
                }

    with open(cache_file, "w") as f:
        json.dump(tokens, f, indent=2)

    return mcp_headers


def list_tools(client: LlamaStackClient):
    headers = ["identifier", "provider_id", "args", "mcp_endpoint"]
    response = client.toolgroups.list()
    if response:
        table = Table()
        for header in headers:
            table.add_column(header)

        for item in response:
            print(item)
            row = [str(getattr(item, header)) for header in headers]
            table.add_row(*row)
        _log.info(table)


def create_agent(
    model_id: str = "qwen3:8b",
    llama_stack_url: str = "http://localhost:8321",
    mcp_servers: str = "https://mcp.asana.com/sse",
    docling_mcp_url: str = "http://host.containers.internal:8000/sse",
):
    client = LlamaStackClient(base_url=llama_stack_url)
    client.toolgroups.register(
        toolgroup_id="mcp::docling",
        provider_id="model-context-protocol",
        mcp_endpoint={"uri": docling_mcp_url},
    )

    list_tools(client)

    if not check_model_exists(client, model_id):
        _log.error(f"model {model_id} is not existing")
        return None
    else:
        _log.info(f"model {model_id} detected")

    """
    servers = [s.strip() for s in mcp_servers.split(",")]
    mcp_headers = get_and_cache_mcp_headers(servers)
    """

    toolgroup_ids = [docling_mcp_url]
    """
    for server in servers:
        # we cannot use "/" in the toolgroup_id because we have some tech debt from earlier which uses
        # "/" as a separator for toolgroup_id and tool_name. We should fix this in the future.
        group_id = urlparse(server).netloc
        toolgroup_ids.append(group_id)
        client.toolgroups.register(
            toolgroup_id=group_id, mcp_endpoint=dict(uri=server), provider_id="model-context-protocol"
        )
    """

    agent = Agent(
        client=client,
        model=model_id,
        instructions="You are a helpful technical assistant who can use tools when necessary to answer questions.",
        tools=toolgroup_ids,
        extra_headers={},
    )
    """
        extra_headers={
            "X-LlamaStack-Provider-Data": json.dumps(
                {
                    "mcp_headers": mcp_headers,
                }
            ),
        },
    """
    return agent


if __name__ == "__main__":
    agent = create_agent()
