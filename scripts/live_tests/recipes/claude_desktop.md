# Test 1.1 — Claude Desktop + MCP Roots (HITL recipe)

Verifies that a Roots-capable client (Claude Desktop) advertises an
authorized directory to docling-mcp via `notifications/roots/list_changed`,
and that the server's `validate_source()` call honors that authorization
when the user asks Claude to parse a file inside that directory.

## What this verifies

| Claim being checked                                                          | How                                                                                              |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Server consumes the client's Roots capability                                | Server log contains `client allowed-roots refreshed`                                             |
| Conversion succeeds for a path inside the advertised root                    | Claude returns content describing the document                                                   |
| (Optional) Conversion is rejected for a path outside the advertised root     | Server log contains `not under any allowed root` for the rejected source                         |

## Prerequisites

* macOS with Claude Desktop installed
* This repo checked out with the post-merge venv synced
  (`uv sync --frozen --all-extras`)
* Fixtures generated:
  `python scripts/live_tests/make_fixtures.py`

## Steps

### 1. Configure Claude Desktop's MCP server

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`.
Add this under `mcpServers` (merge with existing entries if any):

```json
{
  "mcpServers": {
    "docling": {
      "command": "/Users/<you>/Projects/docling-mcp/.venv/bin/docling-mcp-server",
      "args": ["--transport", "stdio"],
      "env": {
        "DOCLING_CONVERSION_MODE": "local"
      }
    }
  }
}
```

Substitute `<you>` for your home directory username.

Notes:
* No `--allowed-directories` flag — Claude Desktop advertises filesystem
  Roots dynamically, and exercising that path is the point of this test.
* `DOCLING_CONVERSION_MODE=local` is required because the post-2.0
  default is `remote` (Docling Serve API). For a local Mac install we
  want the bundled `LocalDocumentConverter`.
* `--transport stdio` is explicit because the post-2.0 default is
  `streamable-http`, which Claude Desktop does not spawn over.

### 2. Restart Claude Desktop

⌘Q (not just close the window), then reopen.

Settings → Developer → MCP Servers. `docling` should show a green dot.
If it doesn't, look at `~/Library/Logs/Claude/mcp-server-docling.log`
for a startup error.

### 3. Grant Claude Desktop filesystem access to the test directory

Settings → Privacy & Security → File System → Add Folder →
`/tmp/docling_roots_live/allowed`.

This is what Claude Desktop will advertise to the docling server as
a Root via `notifications/roots/list_changed`.

### 4. (Optional) Negative side-test, before granting access

Before step 3, skip ahead to step 5 and run the prompt once. Confirm
that the conversion fails with `not under any allowed root`. This
isolates the Roots refresh in step 3 as the change that authorized
the path — not some other piece of Claude Desktop state.

### 5. Open a fresh conversation and prompt

> Use docling to parse `/tmp/docling_roots_live/allowed/test_doc.pdf`
> and describe what's on the page.

### 6. Tail the server log while the prompt runs

```bash
tail -f ~/Library/Logs/Claude/mcp-server-docling.log
```

If that file doesn't exist, find it:

```bash
find ~/Library/Logs/Claude -name "*docling*" -o -name "*mcp*"
```

## What to capture

* The text Claude returned in the chat
* The log lines around the conversion — especially anything mentioning:
  * `client allowed-roots refreshed` (this is the signal the Roots flow fired)
  * `Processing document from source: /tmp/docling_roots_live/allowed/test_doc.pdf`
  * Any `Error converting document` or `not under any allowed root`
* Any errors in the chat or the log

## PASS / FAIL

| | PASS | FAIL |
|---|---|---|
| Server log shows `client allowed-roots refreshed` | yes | no |
| Conversion completes (Claude describes the document content) | yes | no |
| Optional: outside-of-root path rejected with `not under any allowed root` | yes | no |
