"""This module initializes and runs the Docling MCP server."""

import enum
import sys
from typing import Annotated

import typer

from docling_mcp.logger import setup_logger
from docling_mcp.shared import allowed_roots, mcp


def _crumb(step: str) -> None:
    """Emit a flushed stderr breadcrumb.

    Visible in Claude Desktop's log immediately, regardless of Python
    logging buffer state.
    """
    print(f"[docling-mcp-boot] {step}", file=sys.stderr, flush=True)


app = typer.Typer()


class ToolGroups(str, enum.Enum):
    """List of available toolsgroups."""

    CONVERSION = "conversion"
    GENERATION = "generation"
    MANIPULATION = "manipulation"
    LLAMA_INDEX_RAG = "llama-index-rag"
    LLAMA_STACK_RAG = "llama-stack-rag"
    LLAMA_STACK_IE = "llama-stack-ie"


class TransportType(str, enum.Enum):
    """List of available protocols."""

    STDIO = "stdio"
    SSE = "sse"
    STREAMABLE_HTTP = "streamable-http"


_DEFAULT_TOOLS = [ToolGroups.CONVERSION, ToolGroups.GENERATION, ToolGroups.MANIPULATION]


@app.command()
def main(
    transport: TransportType = TransportType.STDIO,
    host: str = "localhost",
    port: int = 8000,
    tools: Annotated[
        list[ToolGroups] | None,
        typer.Argument(
            help=f"Tools to be loaded in the server. The default list is {', '.join(_DEFAULT_TOOLS)}"
        ),
    ] = None,
    allowed_directories: Annotated[
        list[str] | None,
        typer.Option(
            "--allowed-directories",
            help=(
                "Filesystem directories the server is allowed to read from. "
                "Used as a fallback when the client does not advertise the MCP "
                "Roots capability. When the client sends roots, those replace "
                "this list at runtime."
            ),
        ),
    ] = None,
    preload_models: Annotated[
        bool,
        typer.Option(
            "--preload-models",
            help=(
                "Eagerly initialize the Docling DocumentConverter at server "
                "startup so the first conversion tool call doesn't pay the "
                "cold-start cost (PyTorch import, OCR model load, layout "
                "detector init). Adds 30-120s to server boot but lets the "
                "first conversion return inside the MCP client's per-request "
                "timeout (Claude Desktop's spawn timeout is more generous "
                "than its per-request one). Recommended for interactive use."
            ),
        ),
    ] = False,
) -> None:
    """Initialize and run the Docling MCP server."""
    _crumb(f"main() entered (transport={transport.value}, preload={preload_models})")
    # Create a default project logger
    logger = setup_logger()
    _crumb("logger configured")

    if tools is None:
        tools = [*_DEFAULT_TOOLS]

    if ToolGroups.CONVERSION in tools:
        logger.info("loading conversion tools...")
        import docling_mcp.tools.conversion

    if ToolGroups.GENERATION in tools:
        logger.info("loading generation tools...")
        import docling_mcp.tools.generation

    if ToolGroups.MANIPULATION in tools:
        logger.info("loading manipulation tools...")
        import docling_mcp.tools.manipulation

    if ToolGroups.LLAMA_INDEX_RAG in tools:
        logger.info("loading Llama Index RAG tools...")
        import docling_mcp.tools.llama_index.milvus_rag

    if ToolGroups.LLAMA_STACK_RAG in tools:
        logger.info("loading Llama Stack RAG tools...")
        import docling_mcp.tools.llama_stack.rag

    if ToolGroups.LLAMA_STACK_IE in tools:
        logger.info("loading Llama Stack Structured Output tools...")
        import docling_mcp.tools.llama_stack.structured_output

    # Seed the static allowed-roots set from --allowed-directories.
    # Client-sent Roots will replace this set at runtime when present.
    if allowed_directories:
        allowed_roots.set_static_roots(allowed_directories)
        logger.info(f"static allowed-directories: {allowed_directories}")
    else:
        logger.info(
            "no --allowed-directories given; relying on MCP Roots from client "
            "or running unconstrained for backward compatibility"
        )

    # Wire the roots notification handlers onto the FastMCP server.
    from docling_mcp._roots_wiring import install_roots_handlers

    install_roots_handlers()

    # Optionally pay the Docling cold-start cost at boot time. We do this
    # in a BACKGROUND thread so the MCP server can complete its handshake
    # with Claude Desktop fast — the client's spawn timeout would otherwise
    # kill us mid-load. The first convert tool call may then block on the
    # warm-up's lru_cache lock if it arrives before preload completes, but
    # at least the server is reachable from the start.
    # See docling-project/docling-mcp#98.
    if preload_models and ToolGroups.CONVERSION in tools:
        _crumb("preload: scheduling background DocumentConverter warm-up")

        import threading

        def _warmup() -> None:
            _crumb("preload thread: started")
            try:
                from docling_mcp.tools.conversion import _get_converter

                _crumb("preload thread: calling _get_converter()")
                converter = _get_converter()
                _crumb(
                    "preload thread: _get_converter() returned; initializing pipelines"
                )
                # DocumentConverter.__init__ is lazy; pipelines (and the
                # underlying models) load on first .convert(). Force load
                # here via initialize_pipeline() if Docling exposes it,
                # otherwise fall through (preload becomes partial — still
                # helpful because import side-effects are paid).
                try:
                    from docling.datamodel.base_models import InputFormat

                    if hasattr(converter, "initialize_pipeline"):
                        _crumb("preload thread: initialize_pipeline(PDF)")
                        converter.initialize_pipeline(InputFormat.PDF)
                        _crumb("preload thread: initialize_pipeline(IMAGE)")
                        converter.initialize_pipeline(InputFormat.IMAGE)
                    else:
                        _crumb(
                            "preload thread: initialize_pipeline() not "
                            "available on this Docling version; partial "
                            "warm-up only"
                        )
                except Exception as e:
                    _crumb(
                        f"preload thread: pipeline init raised {e!r}; "
                        "falling back to lazy load"
                    )
                _crumb("preload thread: DocumentConverter READY")
            except Exception as e:
                _crumb(f"preload thread: FAILED with {e!r}")

        threading.Thread(target=_warmup, daemon=True, name="docling-preload").start()
        _crumb("preload: background thread launched; proceeding to mcp.run()")

    # Initialize and run the server
    _crumb("about to call mcp.run() — server will block here serving MCP")
    logger.info("starting up Docling MCP-server ...")
    mcp.settings.host = host
    mcp.settings.port = port
    mcp.run(transport=transport.value)


if __name__ == "__main__":
    main()
