"""Utility functions for examples of building agents with Docling MCP and Llama Stack.

This module takes some implementations from [Llama Stack Demos](https://github.com/opendatahub-io/llama-stack-demos/tree/main/demos/rag_agentic).
"""

import json
from json import JSONDecodeError

from rich.pretty import pprint
from termcolor import cprint


def step_printer(steps):
    """
    Print the steps of an agent's response in a formatted way.
    Note: stream need to be set to False to use this function.
    Args:
    steps: List of steps from an agent's response.
    """
    for i, step in enumerate(steps):
        step_type = type(step).__name__
        print("\n" + "-" * 10, f"📍 Step {i + 1}: {step_type}", "-" * 10)
        if step_type == "ToolExecutionStep":
            print("🔧 Executing tool...")
            try:
                pprint(json.loads(step.tool_responses[0].content))
            except (TypeError, JSONDecodeError):
                # tool response is not a valid JSON object
                pprint(step.tool_responses[0].content)
        else:
            if step.api_model_response.content:
                print("🤖 Model Response:")
                cprint(f"{step.api_model_response.content}\n", "magenta")
            elif step.api_model_response.tool_calls:
                tool_call = step.api_model_response.tool_calls[0]
                print("🛠️ Tool call Generated:")
                cprint(
                    f"Tool call: {tool_call.tool_name}, "
                    f"Arguments: {json.loads(tool_call.arguments_json)}",
                    "magenta",
                )
    print("=" * 10, "Query processing completed", "=" * 10, "\n")


def user_printer(query: str) -> None:
    print("👤 User Query:")
    cprint(query, "cyan")
