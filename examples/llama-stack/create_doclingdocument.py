import json
import logging

import httpx
from pathlib import Path

from rich.table import Table

from urllib.parse import urlparse

from llama_stack_client import Agent, AgentEventLogger, LlamaStackClient
from llama_stack_client.lib import get_oauth_token_for_mcp_server

from llama_stack_client.lib.agents.react.agent import ReActAgent

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


def get_toolgroup_ids(client: LlamaStackClient):
    toolgroup_ids = []

    response = client.toolgroups.list()
    if response:
        for item in response:
            toolgroup_ids.append(item.provider_resource_id)

    return toolgroup_ids


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

    if check_model_exists(client, model_id):
        _log.info(f"model {model_id} detected")
    else:
        _log.error(f"model {model_id} is not existing")
        return None

    toolgroup_ids = get_toolgroup_ids(client)

    agent = Agent(
        client=client,
        model=model_id,
        instructions="You are a helpful technical assistant who can use tools when necessary to answer questions.",
        tools=toolgroup_ids,
        extra_headers={},
    )

    return agent


def create_react_agent(
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

    if check_model_exists(client, model_id):
        _log.info(f"model {model_id} detected")
    else:
        _log.error(f"model {model_id} is not existing")
        return None

    toolgroup_ids = get_toolgroup_ids(client)

    agent = ReActAgent(
        client=client,
        model=model_id,
        instructions="You are a helpful technical assistant who can use tools when necessary to answer questions.",
        tools=toolgroup_ids,
        extra_headers={},
    )

    return agent


def run(agent: Agent, user_input: str, session_name: str, stream: bool = True):
    session_id = agent.create_session(session_name)

    response = agent.create_turn(
        session_id=session_id,
        messages=[{"role": "user", "content": user_input}],
        stream=stream,
    )

    if stream:
        for log in AgentEventLogger().log(response):
            log.print()


if __name__ == "__main__":
    agent = create_agent()

    session_name = "test"
    user_input = """Please write a DoclingDocument using the tools. The
    DoclingDocument should be created by consecutively add title, section-headers,
    paragraphs, lists and tables. The DoclingDocument should discuss polymers
    for food-packaging and how WVTR is affected by thickness."""

    run(agent, session_name=session_name, user_input=user_input, stream=True)
