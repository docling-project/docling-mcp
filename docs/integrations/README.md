# Integrating Docling MCP into Clients

Docling MCP can be easily integrated into MCP-compatible clients using standard configuration methods.

## Claude for Desktop

[Claude for Desktop](https://claude.ai/download) supports integration via the `claude_desktop_config.json` file (located at `~/Library/Application Support/Claude/claude_desktop_config.json` in MacOS). By adding the appropriate Docling MCP endpoint and parameters to this configuration, Claude Desktop can connect to and utilize Docling MCP’s functionality with minimal effort. You can find an example of those details [here](claude_desktop_config.json).

Once installed, extend Claude for Desktop so that it can read from your computer's file system, by following the [For Claude Desktop Users](https://modelcontextprotocol.io/quickstart/user) tutorial.


## LM Studio

[LM Studio](https://lmstudio.ai/) supports MCP tools and allows to run your agentic workloads completely locally. The configuration is done by editing the `mcp.json` file, or with the convenience button below.

[![Add MCP Server docling to LM Studio](https://files.lmstudio.ai/deeplink/mcp-install-light.svg)](https://lmstudio.ai/install-mcp?name=docling&config=eyJjb21tYW5kIjoidXZ4IiwiYXJncyI6WyItLWZyb209ZG9jbGluZy1tY3AiLCJkb2NsaW5nLW1jcC1zZXJ2ZXIiXX0%3D)


```json
{
  "mcpServers": {
    "docling": {
      "command": "uvx",
      "args": [
        "--from=docling-mcp",
        "docling-mcp-server"
      ]
    }
  }
} 
```
