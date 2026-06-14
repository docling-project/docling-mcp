"""Live acceptance test 1.3: --allowed-directories static fallback.

Drives the real docling-mcp server over stdio WITHOUT advertising the
client-side Roots capability, so the server uses the static
--allowed-directories set seeded from the CLI flag.

PASS criteria:
    CALL 1 (path inside allowed root)  -> isError=False, content has "document_key"
    CALL 2 (path outside allowed root) -> isError=True, content has
                                          "not under any allowed root"

Usage:
    python scripts/live_tests/test_3_static_roots.py
    python scripts/live_tests/test_3_static_roots.py --workdir /some/other/path
    python scripts/live_tests/test_3_static_roots.py --server-cmd /path/to/docling-mcp-server

Exit code:
    0 if both calls behave as expected
    1 otherwise

Artifacts written under <workdir>:
    server_stderr.log    — full server stderr (Docling logs + roots breadcrumbs)
    result_test_3.json   — structured PASS/FAIL summary
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import timedelta
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def _content_text(result) -> str:
    """Join all TextContent blocks in a CallToolResult into a single string."""
    parts = []
    for c in result.content:
        parts.append(c.text if hasattr(c, "text") else repr(c))
    return "\n".join(parts)


def _default_server_cmd() -> str:
    """Resolve the docling-mcp-server entry point alongside the current Python."""
    return str(Path(sys.executable).parent / "docling-mcp-server")


async def run(workdir: Path, server_cmd: str) -> int:
    """Drive the server through the two acceptance calls; return 0 on PASS."""
    allowed = workdir / "allowed"
    forbidden = workdir / "forbidden"
    pdf_inside = allowed / "test_doc.pdf"
    pdf_outside = forbidden / "test_doc.pdf"
    stderr_log = workdir / "server_stderr.log"
    result_log = workdir / "result_test_3.json"

    for p in (pdf_inside, pdf_outside):
        if not p.exists():
            print(f"ERROR: fixture missing: {p}", file=sys.stderr)
            print("Run: python scripts/live_tests/make_fixtures.py", file=sys.stderr)
            return 2

    # Post-2.0 server defaults: transport is streamable-http and conversion mode
    # is remote. For stdio + local conversion we must pass --transport stdio
    # explicitly and set DOCLING_CONVERSION_MODE=local in the env.
    test_env = os.environ.copy()
    test_env["DOCLING_CONVERSION_MODE"] = "local"
    server = StdioServerParameters(
        command=server_cmd,
        args=["--transport", "stdio", "--allowed-directories", str(allowed)],
        env=test_env,
    )

    summary: dict = {
        "server_command": server_cmd,
        "server_args": server.args,
        "env_override": {"DOCLING_CONVERSION_MODE": "local"},
        "allowed_root": str(allowed),
        "forbidden_root": str(forbidden),
        "calls": [],
        "verdict": None,
    }

    print(f"== launching: {server_cmd} {' '.join(server.args)}")
    print("== env: DOCLING_CONVERSION_MODE=local")
    print(f"== stderr -> {stderr_log}")

    with stderr_log.open("w") as errlog:
        async with stdio_client(server, errlog=errlog) as (read, write):
            async with ClientSession(read, write) as session:
                init_result = await session.initialize()
                summary["server_name"] = init_result.serverInfo.name
                summary["server_version"] = init_result.serverInfo.version
                summary["client_capabilities_sent"] = {
                    "roots": False,
                    "sampling": False,
                }
                print(
                    f"== initialized: server={init_result.serverInfo.name} "
                    f"{init_result.serverInfo.version}"
                )
                print(
                    "== client did NOT advertise 'roots' capability "
                    "(triggers --allowed-directories code path)"
                )

                print(f"\n== CALL 1 (inside) source={pdf_inside}")
                inside_call: dict = {
                    "label": "inside_allowed_root",
                    "source": str(pdf_inside),
                    "expected": "success",
                }
                try:
                    r1 = await session.call_tool(
                        "convert_document_into_docling_document",
                        {"source": str(pdf_inside)},
                        read_timeout_seconds=timedelta(seconds=600),
                    )
                    inside_call["isError"] = bool(r1.isError)
                    inside_call["content_text"] = _content_text(r1)
                    inside_call["structuredContent"] = r1.structuredContent
                    print(f"   isError = {r1.isError}")
                    print(f"   content = {_content_text(r1)[:400]}")
                except Exception as e:
                    inside_call["isError"] = True
                    inside_call["error_class"] = type(e).__name__
                    inside_call["error_text"] = str(e)
                    print(f"   raised {type(e).__name__}: {e}")
                summary["calls"].append(inside_call)

                print(f"\n== CALL 2 (outside) source={pdf_outside}")
                outside_call: dict = {
                    "label": "outside_allowed_root",
                    "source": str(pdf_outside),
                    "expected": "PermissionError",
                }
                try:
                    r2 = await session.call_tool(
                        "convert_document_into_docling_document",
                        {"source": str(pdf_outside)},
                        read_timeout_seconds=timedelta(seconds=60),
                    )
                    outside_call["isError"] = bool(r2.isError)
                    outside_call["content_text"] = _content_text(r2)
                    outside_call["structuredContent"] = r2.structuredContent
                    print(f"   isError = {r2.isError}")
                    print(f"   content = {_content_text(r2)[:400]}")
                except Exception as e:
                    outside_call["isError"] = True
                    outside_call["error_class"] = type(e).__name__
                    outside_call["error_text"] = str(e)
                    print(f"   raised {type(e).__name__}: {e}")
                summary["calls"].append(outside_call)

    c1, c2 = summary["calls"]
    c1_ok = not c1.get("isError") and "document_key" in (c1.get("content_text") or "")
    c2_ok = c2.get("isError") and (
        "not under any allowed root"
        in (c2.get("content_text") or c2.get("error_text") or "")
    )
    summary["verdict"] = "PASS" if (c1_ok and c2_ok) else "FAIL"
    summary["inside_ok"] = c1_ok
    summary["outside_ok"] = c2_ok

    print(f"\n== VERDICT: {summary['verdict']} (inside_ok={c1_ok}, outside_ok={c2_ok})")

    result_log.write_text(json.dumps(summary, indent=2, default=str))
    print(f"== full result -> {result_log}")
    print(f"== server stderr -> {stderr_log}")
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("/tmp/docling_roots_live"),
        help="Directory containing allowed/ and forbidden/ fixtures",
    )
    parser.add_argument(
        "--server-cmd",
        type=str,
        default=_default_server_cmd(),
        help="Path to docling-mcp-server entry point (default: alongside current Python)",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(run(args.workdir, args.server_cmd)))
