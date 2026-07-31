# Migration Guide

## docling-mcp v3 — MCP Python SDK v2

`docling-mcp` v3 migrates from MCP Python SDK v1 to v2. This is the only
breaking change in this release; the docling-mcp tool and prompt APIs are
unchanged.

### Who is affected

Users who run docling-mcp **purely as a standalone server** (e.g. via
`uvx` or a container) and whose MCP client application connects to it over
stdio or HTTP are **not affected** — the server behaviour, CLI flags, and
all tool names are identical across v2 and v3.

Users who are affected:

- Projects that **embed docling-mcp as a library** and import from the MCP
  SDK alongside it — the shared `mcp` dependency will resolve to v2.
- Projects with a **custom MCP client** built on `mcp<2.0.0` that
  programmatically connects to a docling-mcp server — upgrading the server
  to v3 pulls in SDK v2, whose wire protocol changes may affect your client.

### Compatibility table

| docling-mcp version | MCP Python SDK | Notes |
|---|---|---|
| `>=3.0.0` | `mcp>=2.0.0` | current line |
| `>=2.0.0,<3.0.0` (latest: `v2.2.0`) | `mcp>=1.9.4,<2.0.0` | maintenance only |

`v2.2.0` is the last release on the v2 line. It includes a guard
(`mcp[cli]>=1.9.4,<2.0.0`) that was added specifically to prevent
accidental resolution to SDK v2. If you are on `v2.2.0` and see that guard
mentioned in the [changelog](CHANGELOG.md), this is why.

### Staying on v2 (MCP SDK v1-compatible)

If your MCP client application has not yet migrated to MCP SDK v2, pin
docling-mcp to the v2 line:

```bash
pip install "docling-mcp<3.0.0"
```

Or in your `pyproject.toml`:

```toml
dependencies = [
    "docling-mcp>=2.0.0,<3.0.0",
]
```

The v2 line receives critical fixes but no new features.

### Upgrading to v3

Install the latest release:

```bash
pip install --upgrade docling-mcp
```

If your project imports from the MCP SDK directly alongside docling-mcp,
follow the [MCP Python SDK v2 migration guide](https://py.sdk.modelcontextprotocol.io/migration/)
for your own code. The most common changes are:

| v1 | v2 |
|---|---|
| `from mcp.server.fastmcp import FastMCP` | `from mcp.server.mcpserver import MCPServer` |
| `from mcp.server.fastmcp import Context` | `from mcp.server.mcpserver import Context` |
| `from mcp.shared.exceptions import McpError` | `from mcp.shared.exceptions import MCPError` |
| `McpError(ErrorData(code=..., message=...))` | `MCPError(code, message)` |
| `result.isError` | `result.is_error` |
| `result.structuredContent` | `result.structured_content` |
| `tool.inputSchema` | `tool.input_schema` |

---

## docling-mcp v2 — remote/local hybrid architecture

`docling-mcp` v2 introduced a hybrid architecture with support for both
remote API and local conversion modes.

- **Base package** is now ~50 MB (down from ~500 MB), with no model
  downloads required for the default remote mode.
- **Remote mode** (default): uses Docling Serve for scalable cloud-based
  conversion. Set `DOCLING_SERVICE_URL` and optionally
  `DOCLING_SERVICE_API_KEY`.
- **Local mode**: install the `[local]` extra for offline conversion.
  Set `DOCLING_CONVERSION_MODE=local`.
- **Hybrid mode**: install `[local]` and set
  `DOCLING_FALLBACK_TO_LOCAL=true` to fall back to local conversion when
  the remote service is unavailable.

### Upgrading from v1

Replace `pip install docling-mcp` with one of:

```bash
# Remote mode (lightweight, default)
pip install docling-mcp
export DOCLING_SERVICE_URL=https://your-docling-service.example.com

# Local mode (full, offline)
pip install docling-mcp[local]
export DOCLING_CONVERSION_MODE=local
```
