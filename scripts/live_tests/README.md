# Live acceptance tests for the MCP Roots integration (PR #95)

This directory contains end-to-end reproducer scripts and recipes for
the three acceptance criteria in PR #95 that go beyond the unit tests
in `tests/test_roots.py`. They verify that a real docling-mcp server,
running against a real Docling pipeline, honors the MCP Roots protocol
in three scenarios:

| # | Scenario                          | Driver           | Reproducer                              |
| - | --------------------------------- | ---------------- | --------------------------------------- |
| 1.1 | Roots-capable client (Claude Desktop) advertises a directory; conversion of a PDF in that directory succeeds | human            | [`recipes/claude_desktop.md`](recipes/claude_desktop.md) |
| 1.2 | Roots-capable client (Claude Code) advertises its cwd; conversion of a PNG with table structure succeeds | human            | [`recipes/claude_code.md`](recipes/claude_code.md) |
| 1.3 | Static `--allowed-directories` fallback for non-Roots-capable clients; inside-path passes, outside-path rejected | programmatic | [`test_3_static_roots.py`](test_3_static_roots.py) |

Tests 1.1 and 1.2 require a human-driven client and are documented as
recipes. Test 1.3 is fully programmatic and can be re-run on any
checkout in one command.

## Why these aren't pytest tests

* They take 1–2 minutes (real Docling pipeline + model load).
* They require local Docling models, which CI does not provision.
* `DOCLING_CONVERSION_MODE=local` and `--transport stdio` must be set
  explicitly because the post-2.0 server defaults are
  `remote` and `streamable-http`.

CI runs `tests/test_roots.py` (60 unit tests covering the registry,
path utils, and notification handlers). The live tests in this
directory are a layer above that — they prove the wiring composes
end-to-end against a real client + real Docling.

## Prerequisites (all three tests)

1. Repo checked out, on the branch under test (e.g. `feat/mcp-roots-protocol`).
2. Venv synced with all extras:
   ```bash
   uv sync --frozen --all-extras
   ```
3. Fixtures generated:
   ```bash
   python scripts/live_tests/make_fixtures.py
   ```
   Writes a 1-page raster PDF, a PNG with a 3x3 table, and a copy of
   the PDF in a non-allowed dir, under `/tmp/docling_roots_live/`
   (override with `--workdir /some/other/path`).

## Run Test 1.3 (programmatic, ~1 minute)

```bash
source .venv/bin/activate
python scripts/live_tests/test_3_static_roots.py
```

`PASS` means:

* Call 1 (a PDF inside `/tmp/docling_roots_live/allowed/`) returns
  `isError=False` and the response includes a `document_key`.
* Call 2 (the same PDF copied into `/tmp/docling_roots_live/forbidden/`)
  returns `isError=True` and the error message contains
  `not under any allowed root`.

Artifacts written:

* `/tmp/docling_roots_live/server_stderr.log` — full server stderr
* `/tmp/docling_roots_live/result_test_3.json` — structured verdict

## Run Tests 1.1 and 1.2 (HITL)

Each recipe is a self-contained markdown walkthrough — configure the
client, restart it, issue a prompt, capture the response and the
server log. The recipes list `PASS` / `FAIL` criteria so the human
driver can decide deterministically.

* [`recipes/claude_desktop.md`](recipes/claude_desktop.md) — Test 1.1
* [`recipes/claude_code.md`](recipes/claude_code.md) — Test 1.2

These deliberately do not pre-specify "expected output" — the only
PASS / FAIL contract is the structural look-fors in the server log
(`client allowed-roots refreshed`, `Processing document from source:`,
`not under any allowed root`). What Claude says in the chat is
context-dependent on the model and prompt, and asserting on it
would be brittle.

## What the look-fors mean

When you tail the server log during any of these tests, the
signals that prove the Roots flow is exercised correctly are:

| Log line                                              | Means                                                                       |
| ----------------------------------------------------- | --------------------------------------------------------------------------- |
| `static allowed-roots seeded: [...]`                  | Server received `--allowed-directories` and seeded the static set (Test 1.3) |
| `client allowed-roots refreshed (N entries): [...]`   | Client sent `notifications/roots/list_changed` and the registry replaced the static set with N client-provided roots (Tests 1.1 / 1.2) |
| `no --allowed-directories given; relying on MCP Roots from client or running unconstrained for backward compatibility` | Server is in fallback / unconstrained mode (NOT what you want during a live test) |
| `Processing document from source: <path>`             | `validate_source()` passed and the converter is running                     |
| `Error converting document: <path>` followed by `PermissionError: path '...' is not under any allowed root` | `validate_source()` correctly rejected an outside-root path                 |

## Files

* `make_fixtures.py` — PIL-based fixture generator (1-page raster PDF
  + PNG with a 3x3 table). Idempotent — safe to re-run.
* `test_3_static_roots.py` — spawns a real `docling-mcp-server` over
  stdio with `--allowed-directories`, drives two tool calls via the
  `mcp` SDK, asserts the structural PASS criteria, writes a verdict
  JSON and the server stderr log.
* `recipes/claude_desktop.md` — Test 1.1 HITL walkthrough.
* `recipes/claude_code.md` — Test 1.2 HITL walkthrough.
