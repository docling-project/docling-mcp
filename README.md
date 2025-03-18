# Docling MCP: making docling agentic 

[![PyPI version](https://img.shields.io/pypi/v/docling-mcp)](https://pypi.org/project/docling-mcp/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/docling-mcp)](https://pypi.org/project/docling-mcp/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Pydantic v2](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pydantic/pydantic/main/docs/badge/v2.json)](https://pydantic.dev)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![License MIT](https://img.shields.io/github/license/docling-mcp-project/docling-mcp)](https://opensource.org/licenses/MIT)
[![PyPI Downloads](https://static.pepy.tech/badge/docling-mcp/month)](https://pepy.tech/projects/docling-mcp)

A document processing service using the Docling-MCP library and MCP (Message Control Protocol) for tool integration.

## Overview

Docling MCP is a service that provides tools for document conversion, processing and generation. It uses the Docling library to convert PDF documents into structured formats and provides a caching mechanism to improve performance. The service exposes functionality through a set of tools that can be called by client applications.

## Features

- PDF document conversion to structured JSON format
- Local document caching for improved performance
- Support for local files and URLs as document sources
- Memory management for handling large documents
- Logging system for debugging and monitoring

## Getting started

Expose the tools of Docling by running,

```sh
uv run ./docling_mcp/server.py
```

