#!/usr/bin/env python3

import os
import json
import yaml
import logging
import sys
from typing import Optional, Literal, List, Dict, Any
from pathlib import Path
from pydantic import BaseModel, Field, validator
from smolagents import (
    ToolCallingAgent,
    CodeAgent,
    MCPClient,
    ToolCollection,
    TransformersModel,
    LiteLLMModel,
    Tool,
)
from mcp import StdioServerParameters
from packaging import version

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MCPConfig(BaseModel):
    """Configuration for MCP server connection."""

    connection_type: Literal["stdio", "sse"] = Field(
        default="sse", description="Type of MCP connection"
    )
    server_url: str = Field(
        default="http://localhost:8000/sse", description="URL for SSE MCP server"
    )
    command: str = Field(
        default="mcp-server-lls", description="Command to start stdio MCP server"
    )
    args: List[str] = Field(
        default_factory=list, description="Arguments for stdio MCP server"
    )


class ModelConfig(BaseModel):
    """Configuration for the LLM model."""

    type: Literal["transformers", "ollama"] = Field(
        default="transformers", description="Type of model to use"
    )
    model_id: str = Field(
        default="HuggingFaceTB/SmolLM2-1.7B-Instruct", description="Model identifier"
    )
    device: str = Field(default="cuda", description="Device to run on (cuda/cpu)")
    torch_dtype: str = Field(default="auto", description="Torch dtype for model")
    ollama_base_url: str = Field(
        default="http://localhost:11434", description="Base URL for Ollama server"
    )


class PlanningConfig(BaseModel):
    """Configuration for agent behavior."""

    max_steps: int = Field(default=10, description="Maximum steps for agent execution")
    verbosity_level: int = Field(
        default=2, description="Verbosity level (0=quiet, 1=normal, 2=verbose)"
    )
    planning_interval: int = Field(
        default=3, description="Re-planning interval for ToolCallingAgent"
    )
    system_prompt_style: Literal["default", "detailed", "minimal"] = Field(
        default="default", description="Style of system prompt"
    )


class AgentConfig(BaseModel):
    """Complete configuration for Smolagents with MCP."""

    mcp_config: MCPConfig = MCPConfig()
    model_config: ModelConfig = ModelConfig()
    planning_config: PlanningConfig = PlanningConfig()

    stream_output: bool = Field(
        default=False, description="Stream the output of the Agent"
    )

    def do_streaming(self, stream_output: bool = True) -> "AgentConfig":
        self.stream_output = stream_output
        return self

    def with_model_config(self, model_config: ModelConfig) -> "AgentConfig":
        """Return a new AgentConfig with updated model_config."""
        return self.model_copy(update={"model_config": model_config})

    def with_mcp_config(self, mcp_config: MCPConfig) -> "AgentConfig":
        """Return a new AgentConfig with updated mcp_config."""
        return self.model_copy(update={"mcp_config": mcp_config})

    @classmethod
    def from_file(cls, path: Path) -> "AgentConfig":
        """Load configuration from JSON or YAML file."""
        logger.info(f"Loading configuration from {path}")
        with open(path, "r") as f:
            if path.suffix in [".yaml", ".yml"]:
                import yaml

                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        return cls(**data)

    def save(self, path: Path):
        """Save configuration to file."""
        logger.info(f"Saving configuration to {path}")
        data = self.dict()
        with open(path, "w") as f:
            if path.suffix in [".yaml", ".yml"]:
                import yaml

                yaml.dump(data, f, default_flow_style=False)
            else:
                json.dump(data, f, indent=2)


from pathlib import Path
import smolagents
from smolagents.models import Model
from smolagents.tools import Tool
from smolagents.agents import PromptTemplates, populate_template


class DoclingToolCallingAgent(ToolCallingAgent):
    def __init__(
        self,
        agent_type: str,
        tools: list[Tool],
        model: Model,
        prompt_templates: PromptTemplates | None = None,
        planning_interval: int | None = None,
        stream_outputs: bool = False,
        max_tool_threads: int | None = None,
        **kwargs,
    ):
        logger.info("in DoclingToolCallingAgent.__init__")
        prompt_templates = prompt_templates or self._init_prompt_templates()

        super().__init__(
            tools=tools,
            model=model,
            # prompt_templates=None,
            prompt_templates=prompt_templates,
            planning_interval=planning_interval,
            stream_outputs=stream_outputs,
            max_tool_threads=max_tool_threads,
            **kwargs,
        )

    def _init_prompt_templates(self) -> PromptTemplates:
        """Initialize prompt templates based on smolagents version."""
        current_version = version.parse(smolagents.__version__)

        # Define version-specific template mappings
        version_templates = [
            (version.parse("1.21.0"), "toolcalling_agent_latest.yaml"),
            (version.parse("1.20.0"), "toolcalling_agent_v1.20-release.yaml"),
        ]

        # Find the appropriate template file
        chosen_version = None
        template_file = None
        for ver, template in version_templates:
            if current_version >= ver:
                chosen_version = ver
                template_file = template
                break

        if template_file is None:
            # Fallback to oldest supported version
            template_file = "toolcalling_agent_v1.20.0.yaml"
            logger.warning(
                f"No template found for smolagents version {smolagents.__version__}. "
                f"Using fallback: {template_file}"
            )

        file_path = Path(__file__).parent / "resources" / template_file
        with open(file_path, "r") as fr:
            prompt_templates = PromptTemplates(yaml.safe_load(fr))

        return prompt_templates

    def initialize_system_prompt(self) -> str:
        print("in DoclingToolCallingAgent.initialize_system_prompt")
        print("DoclingToolCallingAgent: ", self.tools.values())

        system_prompt = populate_template(
            self.prompt_templates["system_prompt"],
            variables={
                "tools": self.tools,
                "managed_agents": self.managed_agents,
                "custom_instructions": self.instructions,
            },
        )
        return system_prompt


class SmolAgentFactory:
    """Main class to demonstrate local agents with MCP tools."""

    def __init__(self, config: AgentConfig):
        """Initialize with configuration."""
        self.config = config
        print(self.config)

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.logger.info("Initializing LocalSmolagentsWithMCP")

        # Setup local model
        self.model = self._setup_local_model()

        # Setup MCP tools
        self.tools = self._setup_mcp_tools()

        self.logger.info(f"Initialized with {len(self.tools)} MCP tools")
        self.logger.info(
            f"Using model: {self.config.model_config.type} - {self.config.model_config.model_id}"
        )

    def _setup_local_model(self):
        """Setup local model based on configuration."""
        self.logger.info(f"Setting up {self.config.model_config} model")

        if self.config.model_config.type == "ollama":
            self.logger.info(
                f"Connecting to Ollama at {self.config.model_config.ollama_base_url}"
            )
            # return OllamaModel(
            return LiteLLMModel(
                model_id=self.config.model_config.model_id,
                base_url=self.config.model_config.ollama_base_url,
                num_ctx=4
                * 8192,  # ollama default is 2048 which will often fail horribly.
            )
        else:
            self.logger.info(
                f"Loading transformers model: {self.config.model_config.model_id}"
            )
            return TransformersModel(
                model_id=self.config.model_config.model_id,
                device=self.config.model_config.device,
                torch_dtype=self.config.model_config.torch_dtype,
                trust_remote_code=True,
                max_new_tokens=4 * 8192,
            )

    def _setup_mcp_tools(self):
        """Setup MCP tools based on configuration."""
        if self.config.mcp_config.connection_type == "sse":
            server_parameters = {
                "url": self.config.mcp_config.server_url,
                "transport": "sse",
            }
            self.logger.info(
                f"Connecting to MCP server at {self.config.mcp_config.server_url}"
            )
        else:
            server_parameters = StdioServerParameters(
                command=self.config.mcp_config.command,
                args=self.config.mcp_config.args,
                env=os.environ.copy(),
            )
            self.logger.info(
                f"Starting MCP server with command: {self.config.mcp_config.command}"
            )

        try:
            self.mcp_client = MCPClient(server_parameters)
            tools = self.mcp_client.get_tools()

            for i, tool in enumerate(tools):
                self.logger.info(f"tool-{i}:\t {tool}")

            self.logger.info(f"Successfully loaded {len(tools)} tools from MCP server")
            return tools
        except Exception as e:
            self.logger.error(f"Failed to connect to MCP server: {e}")
            raise

    def _get_system_prompt(self, agent_type: str) -> str:
        """Get system prompt based on configuration."""
        prompts = {
            "tool_calling": {
                "default": """You are a helpful document processing assistant.
                You have access to document processing tools via MCP.
                Always think step by step and use the appropriate tools to help the user.
                When processing documents, be thorough and provide detailed analysis.""",
                "detailed": """You are an expert document analysis assistant with access to MCP tools.
                
                Your capabilities include:
                - Converting documents to various formats (markdown, text, etc.)
                - Extracting tables, figures, and structured data
                - Generating summaries and Q&A content
                - Creating new documents with specific formatting
                
                Always:
                1. Understand the user's request completely
                2. Choose the most appropriate tool for the task
                3. Provide detailed, well-structured responses
                4. Handle errors gracefully and suggest alternatives""",
                "minimal": "You are a document processing assistant. Use tools efficiently.",
            },
            "react": {
                "default": """You are a document analysis expert using ReAct reasoning.
                
                For each task:
                1. THINK about what needs to be done
                2. ACT by using the appropriate tool
                3. OBSERVE the results
                4. REPEAT until the task is complete
                
                Always explain your reasoning before taking actions.
                Use the document processing tools to analyze and extract information.""",
                "detailed": """You are an advanced document analysis system using ReAct methodology.
                
                Your approach:
                - REASONING: Carefully analyze what the user needs and break it down into steps
                - ACTION: Select and use the most appropriate tools from your MCP toolkit
                - OBSERVATION: Examine results critically and determine next steps
                - ITERATION: Continue until you've fully addressed the user's request
                
                Available capabilities:
                - Document format conversion (PDF, DOCX, HTML → Markdown)
                - Table and figure extraction with analysis
                - Content summarization and Q&A generation
                - Document creation and structuring
                
                Always provide clear reasoning for your actions and comprehensive results.""",
                "minimal": "Think step-by-step. Use ReAct pattern: Reason, Act, Observe. Be concise.",
            },
        }

        prompt = prompts[agent_type][self.config.planning_config.system_prompt_style]
        self.logger.debug(
            f"Using {self.config.planning_config.system_prompt_style} prompt for {agent_type}"
        )

        return prompt

    def create_tool_calling_agent(self):
        """Create a ToolCallingAgent based on configuration."""
        self.logger.info("Creating DoclingToolCallingAgent")

        """
        print("================================================================================")
        agent = ToolCallingAgent(            
            tools=self.tools,
            model=self.model,
            max_steps=self.config.planning_config.max_steps,
            verbosity_level=self.config.planning_config.verbosity_level,
            planning_interval=self.config.planning_config.planning_interval,
            stream_outputs=True,
            instructions="" #self._get_system_prompt("tool_calling"),
        )

        instructions = agent.initialize_system_prompt()
        logger.info(f"instructions for the agent:\n\n{instructions[0:128]}")

        print("================================================================================")
        """

        agent = DoclingToolCallingAgent(
            agent_type="writing",
            tools=self.tools,
            model=self.model,
            max_steps=self.config.planning_config.max_steps,
            verbosity_level=self.config.planning_config.verbosity_level,
            planning_interval=self.config.planning_config.planning_interval,
            stream_outputs=True,
            instructions="",  # self._get_system_prompt("tool_calling"),
        )

        instructions = agent.initialize_system_prompt()
        logger.info(f"instructions for the agent:\n\n{instructions[0:128]}")

        input("continue?")

        return agent

    def create_react_agent(self):
        """Create a ReAct-style agent based on configuration."""
        self.logger.info("Creating CodeAgent (ReAct)")
        agent = CodeAgent(
            tools=self.tools,
            model=self.model,
            max_steps=self.config.planning_config.max_steps,
            verbosity_level=self.config.planning_config.verbosity_level,
            instructions=self._get_system_prompt("react"),
            stream_outputs=True,
        )
        return agent


def main():
    """Main function to run the demonstrations."""
    print("Local Smolagents with MCP Tools Demo")

    model_config = ModelConfig(
        type="ollama", model_id="ollama/smollm2", device="cpu", torch_dtype="auto"
    )

    agent_config = (
        AgentConfig().with_model_config(model_config=model_config).do_streaming(True)
    )

    factory = SmolAgentFactory(config=agent_config)

    agent = factory.create_tool_calling_agent()

    result = agent.run(
        "Write a short document on the use of polymers for food-packaging."
    )
    print(result)


if __name__ == "__main__":
    main()
