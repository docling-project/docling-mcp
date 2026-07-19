# IBM Bob

[IBM Bob](https://bob.ibm.com) is IBM's agentic SDLC tool, available as Bob IDE and Bob Shell. Both read MCP server definitions from JSON configuration files, so Docling MCP plugs in with a single `mcpServers` entry.

The server process always runs on your machine over stdio. The `DOCLING_CONVERSION_MODE` environment variable decides where document conversion happens: `local` converts in-process, `remote` delegates conversion to a [docling-serve](https://github.com/docling-project/docling-serve) endpoint.

## Prerequisites

- Bob IDE or Bob Shell.
- [uv](https://docs.astral.sh/uv/) on your `PATH`. The examples launch the server with `uvx`.
- Local mode: Python 3.10 or later available to uv, plus disk space for the Docling models downloaded on first conversion.
- Remote mode: a reachable docling-serve endpoint, and an API key if the endpoint requires one.

## Configuration placement

Bob merges MCP servers from two levels:

- **Global**: `~/.bob/settings/mcp.json`, applied to every workspace. Open the Bob panel and select the **MCP** tab to edit it. Bob releases before 2.0.0 used `~/.bob/mcp_settings.json`; Bob migrates that file automatically.
- **Project**: `.bob/mcp.json` in the project root. Bob creates the file if it does not exist. Commit it to share the setup with your team.

When the same server name exists at both levels, the project entry takes precedence.

## Local mode

Add this to your global configuration or to `.bob/mcp.json`:

```json
{
  "mcpServers": {
    "docling": {
      "command": "uvx",
      "args": [
        "--from=docling-mcp[local]==2.1.0",
        "docling-mcp-server",
        "--transport",
        "stdio"
      ],
      "env": {
        "DOCLING_CONVERSION_MODE": "local"
      }
    }
  }
}
```

Two details matter here:

- `--transport stdio` is required. Since v2 the server defaults to `streamable-http`, which Bob's stdio transport cannot talk to.
- The `[local]` extra is required. The base package ships without the conversion models support, and `DOCLING_CONVERSION_MODE=local` fails with an `ImportError` if the extra is missing.

## Remote mode

Conversion runs on a docling-serve endpoint instead of your machine, so the base package is enough:

```json
{
  "mcpServers": {
    "docling": {
      "command": "uvx",
      "args": [
        "--from=docling-mcp==2.1.0",
        "docling-mcp-server",
        "--transport",
        "stdio"
      ],
      "env": {
        "DOCLING_CONVERSION_MODE": "remote",
        "DOCLING_SERVICE_URL": "https://your-docling-serve.example.com"
      }
    }
  }
}
```

`remote` is the default conversion mode; it is set explicitly here for clarity. If your endpoint enforces authentication, add `"DOCLING_SERVICE_API_KEY": "<your-key>"` to the `env` block. Bob stores `env` values literally and does not expand `${VAR}` references, so a key in a committed project `.bob/mcp.json` is exposed verbatim. Keep the credentialed entry in your global configuration instead.

One more environment variable is honored in remote mode: set `DOCLING_FALLBACK_TO_LOCAL=true` to fall back to local conversion when the endpoint is unreachable. The fallback requires the `[local]` extra. The settings module also defines `DOCLING_SERVICE_TIMEOUT` and `DOCLING_SERVICE_MAX_RETRIES`, but in 2.1.0 they are not passed through to the docling-serve client, which uses a fixed 300 second job timeout and 3 HTTP retries.

## Verify the setup

Save the configuration and open the **MCP** tab in the Bob panel. A server named `docling` should appear. Expanding it lists 19 tools with the default tool groups:

- Conversion: `convert_document_into_docling_document`, `convert_directory_files_into_docling_document`, `is_document_in_local_cache`
- Generation: `create_new_docling_document`, `export_docling_document_to_markdown`, `save_docling_document`, `page_thumbnail`, plus the `add_*`, `open_list_*`, and `close_list_*` authoring tools
- Manipulation: `get_overview_of_document_anchors`, `search_for_text_in_document_anchors`, `get_text_of_document_item_at_anchor`, `update_text_of_document_item_at_anchor`, `delete_document_items_at_anchors`

The server also publishes seven MCP prompts, such as `convert_and_summarize`, which Bob surfaces as commands alongside its built-in ones.

Then try a conversion with any PDF in your workspace:

```prompt
Convert /path/to/spec.pdf to a DoclingDocument and return its document-key.
```

Bob asks for approval to run `convert_document_into_docling_document` (see [Tool approval prompts](#every-tool-call-asks-for-approval)) and the tool returns:

```json
{
  "from_cache": false,
  "document_key": "<32-character hex key>"
}
```

The `document_key` is a hash derived from the exact source string, so it differs between machines and between relative and absolute paths. Repeating the prompt returns `"from_cache": true` with the same key. The manipulation and export tools accept the key, so Bob can summarize, search, or edit the document without pasting its full text into the conversation. The cache lives in the server process memory; if Bob restarts the server, convert the document again. If the tool reports that the file cannot be found, pass an absolute path (see [Troubleshooting](#the-server-cannot-find-a-file-you-referenced)).

## Troubleshooting

### First conversion in local mode is slow or times out

On first use, `uvx` resolves and installs the pinned package, and Docling downloads its conversion models during the first conversion. Bob's per-server timeout defaults to 60000 ms (1 minute); it accepts values up to 3600000 and snaps them to fixed steps (5 s, 10 s, 30 s, 1, 2, 5, 10, 30, and 60 minutes). Add for example `"timeout": 600000` to the `docling` server entry before the first run, or warm the package cache outside Bob first:

```sh
uvx --from='docling-mcp[local]==2.1.0' docling-mcp-server --help
```

The model download still happens on the first conversion, so expect that one call to take several minutes. Later conversions reuse the cached models.

### Large PDFs exceed the tool timeout

Raise the `timeout` value on the `docling` server entry, for example to `300000` (5 minutes). In remote mode the docling-serve request additionally has a fixed 300 second job timeout in docling-mcp 2.1.0, independent of Bob's setting. For scanned documents, OCR dominates the conversion time. Prefer converting one large file per tool call rather than pointing `convert_directory_files_into_docling_document` at a directory of large files.

### The server cannot find a file you referenced

The MCP server is a separate process. Bob spawns stdio servers with only the configured `command`, `args`, and `env`, so the server inherits Bob's own working directory, not the workspace root. Use absolute paths in prompts. This applies to remote mode as well: local files are read by the server process on your machine before conversion is delegated.

### `uvx` is not found

Bob IDE launched from the desktop may not inherit your shell `PATH`. Replace `"command": "uvx"` with the absolute path printed by `which uvx`.

### Every tool call asks for approval

That is Bob's default for MCP tools. To pre-approve specific tools, add them to the server's `alwaysAllow` array, for example `"alwaysAllow": ["is_document_in_local_cache", "export_docling_document_to_markdown"]`, or toggle auto-approval per tool from the server's entry in the MCP tab. The related `disabledTools` array hides individual tools from Bob entirely.

## Bob skill

A ready-made skill in [bob/skill/](bob/skill/) teaches Bob to reach for Docling whenever a task references a PDF, DOCX, PPTX, or scanned document: convert first, then work from the structured output by `document-key` instead of pasting raw text.

Install it by copying the folder to `.bob/skills/docling/` in your workspace, or to `~/.bob/skills/docling/` to make it available everywhere:

```sh
mkdir -p .bob/skills/docling
cp docs/integrations/bob/skill/SKILL.md .bob/skills/docling/
```

Bob watches its skills directories and picks up new or edited skills without a restart. When a request matches the skill's `description`, Bob activates it through its `use_skill` tool, asking for approval unless skills are auto-approved in the settings. Skills are available in every mode by default; an optional `groups` frontmatter field restricts a skill to specific modes.

## Example configurations

Ready-to-commit variants of the configuration above, one file per mode, are provided in [bob/](bob/):

- [bob/mcp.local.json](bob/mcp.local.json)
- [bob/mcp.remote.json](bob/mcp.remote.json)

Copy the one you need to `.bob/mcp.json` in your project, or merge its `docling` entry into your global configuration. The remote variant deliberately omits `DOCLING_SERVICE_API_KEY`; if your endpoint needs one, add it to your global configuration rather than the committed project file.
