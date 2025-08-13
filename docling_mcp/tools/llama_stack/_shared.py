"""This module contains functions for working with Llama Stack."""

from functools import lru_cache

from llama_stack_client import LlamaStackClient

from docling_mcp.settings.llama_stack import settings


@lru_cache
def get_llama_stack_client() -> LlamaStackClient:
    client = LlamaStackClient(
        base_url=settings.url,
    )
    return client
