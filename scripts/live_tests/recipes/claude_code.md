# Test 1.2 — Claude Code + MCP Roots (HITL recipe)

Verifies that Claude Code's automatic cwd-derived Roots advertisement
reaches the docling-mcp server and that the server's `validate_source()`
call honors that authorization when the user asks Claude to parse a file
inside that directory.

## What this verifies

| Claim being checked                                                       | How                                                                              |
| ------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| Server consumes Claude Code's cwd as a Root                               | Server log contains `client allowed-roots refreshed`                             |
| Conversion of an image (`test_table.png`) succeeds with table-structure   | Claude returns a description of the 3-row table                                  |
| (Optional) Conversion is rejected when cwd does not cover the file path   | Server log contains `not under any allowed root` for the rejected source         |

`do_table_structure` is True by default in `PdfPipelineOptions`, so no
flag plumbing at the tool layer is required for the table-structure
assertion in PR #95's acceptance criteria.

## Prerequisites

* `claude` CLI installed (Claude Code)
* This repo checked out with the post-merge venv synced
  (`uv sync --frozen --all-extras`)
* Fixtures generated:
  `python scripts/live_tests/make_fixtures.py`

## Steps

### 1. Register the docling MCP server with Claude Code

```bash
claude mcp add docling \
  /Users/<you>/Projects/docling-mcp/.venv/bin/docling-mcp-server \
  -e DOCLING_CONVERSION_MODE=local \
  -- --transport stdio
```

* `-e KEY=VALUE` passes environment variables to the spawned server.
* Everything after `--` is forwarded as server CLI args.
* `DOCLING_CONVERSION_MODE=local` and `--transport stdio` are required
  for the same reasons as Test 1.1 (see `claude_desktop.md`).

### 2. Start a fresh Claude Code session from inside the allowed directory

```bash
cd /tmp/docling_roots_live/allowed
claude
```

The cwd is what Claude Code advertises as a Root, so starting from
inside `allowed/` is what gives the server permission to read the
fixture file.

### 3. Prompt

> Use docling to parse `/tmp/docling_roots_live/allowed/test_table.png`
> and describe the table structure.

### 4. Tail the server log while the prompt runs

The log path varies by Claude Code version. Find it:

```bash
find ~/.claude ~/Library/Logs -name "*docling*log*" 2>/dev/null
```

Tail whichever matches.

### 5. (Optional) Negative side-test

In a separate terminal:

```bash
cd /tmp     # NOT under /tmp/docling_roots_live/allowed
claude
```

Repeat the same prompt. Expect rejection with
`not under any allowed root`. Proves that the cwd-derived Root is
what authorized the file in step 3, not some open fallback.

## What to capture

* Claude Code's response (text in the terminal)
* The log lines around the conversion — same look-fors as Test 1.1:
  * `client allowed-roots refreshed`
  * `Processing document from source: /tmp/docling_roots_live/allowed/test_table.png`
  * Any errors

## PASS / FAIL

| | PASS | FAIL |
|---|---|---|
| Server log shows `client allowed-roots refreshed` with the cwd in the set | yes | no |
| Conversion completes (Claude describes the table) | yes | no |
| Optional: cwd outside `allowed/` rejects the same source path | yes | no |
