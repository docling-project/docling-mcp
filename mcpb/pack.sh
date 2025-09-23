#!/bin/bash

# Python lib
python -m pip install -U pipx --target ./lib

# Assets (logo, screenshots, etc)
mkdir -p assets/
cp ../docs/assets/* assets/

# Pack
npx -y @anthropic-ai/mcpb pack . docling-mcp.mcpb
