"""Example of ACP client prompt on an ACP server."""

import asyncio

from acp_sdk.client import Client
from colorama import Fore


async def run_conversion_workflow() -> None:
    """Run the document conversion workflow by calling the docling_agent."""
    async with Client(base_url="http://localhost:4242") as acp_client:
        run1 = await acp_client.run_sync(
            agent="docling_agent",
            input="Please convert the document at https://arxiv.org/pdf/2408.09869 to markdown and summarize its content.",
        )
        content = run1.output[0].parts[0].content
        print(Fore.LIGHTMAGENTA_EX + content + Fore.RESET)


if __name__ == "__main__":
    asyncio.run(run_conversion_workflow())
