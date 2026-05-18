#!/bin/bash
set -e

# Read version from ../pyproject.toml
VERSION=$(grep -E '^[[:space:]]*version[[:space:]]*=' ../pyproject.toml \
          | head -n1 \
          | sed -E 's/[[:space:]]*version[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/')
echo "Using version: $VERSION"

# Update manifest.json
jq --arg ver "$VERSION" \
   '.version = $ver | .user_config.version.default = $ver' \
   manifest.json > manifest.tmp.json && mv manifest.tmp.json manifest.json

# Python lib
python -m pip install -U pipx --target ./lib

# Assets (logo, screenshots, etc)
mkdir -p assets/
cp ../docs/assets/* assets/

# Pack
npx -y @anthropic-ai/mcpb pack . docling-mcp.mcpb
