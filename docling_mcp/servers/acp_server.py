import os
from collections.abc import AsyncGenerator

from acp_sdk.models import Message
from acp_sdk.server import Context, RunYield, RunYieldResume, Server
from dotenv import load_dotenv
from mcp import StdioServerParameters
from smolagents import (
    LiteLLMModel,
    ToolCallingAgent,
    ToolCollection,
)

load_dotenv()

server = Server()

model = LiteLLMModel(
    model_id=os.getenv("OLLAMA_MODEL"),
    num_ctx=8192,
)

# Outline STDIO stuff to get to MCP Tools
server_parameters = StdioServerParameters(
    command="uv",
    args=["run", "docling_mcp/server.py"],
    env=None,
)


@server.agent()
async def convert_agent(
    input: list[Message], context: Context
) -> AsyncGenerator[RunYield, RunYieldResume]:
    """This is a CodeAgent which supports convering PDF documents."""
    with ToolCollection.from_mcp(
        server_parameters, trust_remote_code=True
    ) as tool_collection:
        agent = ToolCallingAgent(tools=[*tool_collection.tools], model=model)
        prompt = input[0].parts[0].content
        response = agent.run(prompt)

    yield response


if __name__ == "__main__":
    server.run(port=4242)
