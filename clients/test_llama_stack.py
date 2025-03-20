import os

from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types.agent_create_params import AgentConfig
from llama_stack_client.types.shared_params.url import URL
from termcolor import cprint

# Set your model ID
model_id = os.environ["INFERENCE_MODEL"]
client = LlamaStackClient(
    base_url=f"http://localhost:{os.environ.get('LLAMA_STACK_PORT', '8080')}"
)

# Check available providers
available_providers = client.providers.list()
cprint(f"Available providers: {[p.provider_id for p in available_providers]}", "yellow")

# Find the MCP provider
mcp_provider = next(
    (p for p in available_providers if p.provider_id == "model-context-protocol"), None
)
if not mcp_provider:
    cprint(
        "MCP provider not found. Please make sure it's installed and enabled.", "red"
    )
    exit(1)

cprint(f"Using MCP provider: {mcp_provider.provider_id}", "green")

# Try to unregister the toolgroup if it exists
try:
    client.toolgroups.unregister(toolgroup_id="mcp::docling")
    cprint("Unregistered existing toolgroup", "yellow")
except Exception:
    pass  # Ignore if it doesn't exist

# Register MCP tools
try:
    client.toolgroups.register(
        toolgroup_id="mcp::docling",
        provider_id="model-context-protocol",
        mcp_endpoint=URL(uri="http://localhost:8000/sse"),
    )
    cprint("Successfully registered MCP docling toolgroup", "green")
except Exception as e:
    cprint(f"Error registering MCP toolgroup: {e}", "red")
    exit(1)

# Define an agent with MCP toolgroup
agent_config = AgentConfig(
    model=model_id,
    instructions="""You are a helpful assistant with access to tools that can convert documents to markdown.
When asked to convert a document, use the 'convert_document' tool.
You can also extract tables with 'extract_tables' or get images with 'convert_document_with_images'.
Always use the appropriate tool when asked to process documents.""",
    toolgroups=["mcp::docling"],
    input_shields=[],
    output_shields=[],
    enable_session_persistence=False,
    tool_choice="auto",
    tool_prompt_format="python_list",
    max_tool_calls=3,
)

# Create the agent
agent = Agent(client, agent_config)
cprint("Successfully created agent", "green")

# Create a session
session_id = agent.create_session("test-session")
cprint(f"Created session: {session_id}", "green")

# Define the prompt
prompt = "Please convert the document at https://arxiv.org/pdf/2004.07606 to markdown and summarize its content."
cprint(f"User> {prompt}", "green")

try:
    # Create a turn with error handling
    response = agent.create_turn(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        session_id=session_id,
    )

    # Use try-except for each log to handle potential None values
    for log in EventLogger().log(response):
        try:
            if log is not None:
                log.print()
            else:
                cprint("Received None log entry", "yellow")
        except Exception as e:
            cprint(f"Error processing log: {e}", "red")
except Exception as e:
    cprint(f"Error creating turn: {e}", "red")
    import traceback

    traceback.print_exc()
