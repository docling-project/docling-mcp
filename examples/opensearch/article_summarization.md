# Document Summarization with OpenSearch Agentic Search and Docling MCP

This tutorial demonstrates how to use Docling MCP as an external MCP server with OpenSearch's agentic search to automatically fetch, process, and summarize enterprise documents using only the OpenSearch API.s

OpenSearch introduced native [support for external Model Context Protocol (MCP) tools](https://docs.opensearch.org/latest/vector-search/ai-search/agentic-search/mcp-server/) with the release of
OpenSearch 3.0 (May 2025). This guide shows how to create an intelligent document repository where users can ask for summaries in natural language.

## Why This Approach is Powerful

**Traditional approach:** You would need to:
- Run Docling locally with Python code
- Parse each document and understand its structure
- Decide which parts to index (text, tables, images)
- Design mappings for different data types
- Write bulk ingestion scripts
- Handle document updates and versioning

**This approach:** You simply:
- Store metadata (title, URL, category, date) in OpenSearch
- Let Docling MCP handle all document processing on-demand
- Use natural language queries like any other OpenSearch search
- No coding required - just REST API calls

**Key benefit:** You leverage OpenSearch's search power and Docling's document processing power without writing any document processing code. If you're familiar with OpenSearch, you can run agentic search queries exactly as you would normally, and the agent automatically delegates document processing to Docling MCP when needed.

**Note:** This approach is shown for illustration purposes, using an on-demand document processing use case. For large-scale RAG applications on extensive document collections, you may want to consider more robust agentic pipelines like [OpenRAG](https://www.openr.ag/), which combines Docling, OpenSearch, and Langflow to visually create RAG pipelines for agentic search within minutes.

## Use Case: Intelligent Article Repository

Users can ask natural language questions like:
- "Give me a summary of the latest article from the document-processing category"
- "Summarize the most recent transformers paper"
- "What's the latest research in AI?"

The OpenSearch agent will automatically:
1. Query the index to find the latest article in the specified category
2. Use Docling MCP to fetch and convert the PDF from the URL
3. Use Docling MCP to export the document to markdown
4. Generate a summary from the markdown content

## Architecture

```
User Query: "Summarize the latest AI article"
        ↓
OpenSearch Agentic Search Agent
        ↓
    ┌───────────────────────────────────┐
    │  Agent Orchestrates Tools:        │
    │                                   │
    │  1. SearchIndexTool (OpenSearch)  │
    │     → Find latest article         │
    │                                   │
    │  2. convert_document (Docling)    │
    │     → Fetch & convert PDF         │
    │                                   │
    │  3. export_markdown (Docling)     │
    │     → Get text representation     │
    │                                   │
    │  4. LLM generates summary         │
    └───────────────────────────────────┘
        ↓
    Summary Result
```

## Prerequisites

- OpenSearch 3.0+ with ML Commons plugin. This tutorial runs OpenSearch containerized using **Podman**. Refer to [Run OpenSearch in a Docker container](https://docs.opensearch.org/latest/install-and-configure/install-opensearch/docker/) for more details.
- Docling MCP server running
- OpenAI API key (or other LLM provider)

## Step 1: Start Docling MCP Server

Start the Docling MCP server with streamable HTTP transport:

```bash
uvx --from docling-mcp docling-mcp-server --transport streamable-http --port 8000 conversion generation
```

Verify it's running:

```bash
curl http://localhost:8000/mcp
```

## Step 2: Create Articles Index

Create an index to store article metadata including title, PDF URL, category, and publication date:

```json
PUT /articles-index
{
  "settings": {
    "number_of_shards": 1,
    "number_of_replicas": 0
  },
  "mappings": {
    "properties": {
      "article_id": {"type": "keyword"},
      "title": {
        "type": "text",
        "fields": {"keyword": {"type": "keyword"}}
      },
      "pdf_url": {"type": "keyword"},
      "category": {"type": "keyword"},
      "publication_date": {"type": "date"},
      "authors": {"type": "text"},
      "abstract": {"type": "text"},
      "page_count": {"type": "integer"},
      "tags": {"type": "keyword"}
    }
  }
}
```

## Step 3: Index Sample Articles

Add sample articles with metadata. We use real arXiv papers for demonstration:

```json
POST _bulk
{"index": {"_index": "articles-index", "_id": "art1"}}
{"article_id": "art1", "title": "Attention Is All You Need", "pdf_url": "https://arxiv.org/pdf/1706.03762", "category": "transformers", "publication_date": "2017-06-12", "authors": "Vaswani et al.", "abstract": "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms.", "page_count": 15, "tags": ["transformers", "attention", "nlp", "deep-learning"]}
{"index": {"_index": "articles-index", "_id": "art2"}}
{"article_id": "art2", "title": "DocLayNet: A Large Human-Annotated Dataset for Document-Layout Analysis", "pdf_url": "https://arxiv.org/pdf/2206.01062", "category": "document-processing", "publication_date": "2022-06-02", "authors": "Pfitzmann et al.", "abstract": "Accurate document layout analysis is a key requirement for high-quality PDF document conversion.", "page_count": 12, "tags": ["document-ai", "layout", "dataset", "pdf"]}
{"index": {"_index": "articles-index", "_id": "art3"}}
{"article_id": "art3", "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", "pdf_url": "https://arxiv.org/pdf/1810.04805", "category": "transformers", "publication_date": "2018-10-11", "authors": "Devlin et al.", "abstract": "We introduce a new language representation model called BERT.", "page_count": 16, "tags": ["bert", "nlp", "pre-training", "transformers"]}
{"index": {"_index": "articles-index", "_id": "art4"}}
{"article_id": "art4", "title": "Docling Technical Report", "pdf_url": "https://arxiv.org/pdf/2408.09869", "category": "document-processing", "publication_date": "2024-08-19", "authors": "Auer et al.", "abstract": "Docling is an open-source toolkit for document conversion and processing.", "page_count": 9, "tags": ["docling", "document-ai", "pdf", "conversion"]}
```

Refresh the index to make documents searchable:

```json
POST /articles-index/_refresh
```

## Step 4: Register Docling MCP Connector

Register Docling MCP as an external connector. Note that the credential `access_key` is here irrelevant.

```json
POST /_plugins/_ml/connectors/_create
{
  "name": "Docling MCP Connector for Article Summarization",
  "description": "Connector to Docling MCP server for document processing and conversion",
  "version": 1,
  "protocol": "mcp_streamable_http",
  "parameters": {
    "endpoint": "/mcp"
  },
  "credential": {
    "access_key": "docling-mcp-key"
  },
  "url": "http://host.containers.internal:8000/mcp",
  "headers": {
    "Authorization": "Bearer ${credential.access_key}",
    "Content-Type": "application/json"
  }
}
```

**Sample Response:**
```json
{
  "connector_id": "q9wr9JwBjrSrr3-XSvbw"
}
```

Save the `connector_id` for the next steps.

## Step 5: Register LLM Model

Register an OpenAI model (replace `<YOUR_OPENAI_KEY>` with your actual API key):

```json
POST /_plugins/_ml/models/_register
{
  "name": "OpenAI GPT-5 for Article Summarization",
  "function_name": "remote",
  "description": "Model for agentic search with Docling MCP tools for article summarization",
  "connector": {
    "name": "OpenAI Connector",
    "description": "Connector to OpenAI chat model",
    "version": 1,
    "protocol": "http",
    "parameters": {
      "model": "gpt-5"
    },
    "credential": {
      "openAI_key": "<YOUR_OPENAI_KEY>"
    },
    "actions": [
      {
        "action_type": "predict",
        "method": "POST",
        "url": "https://api.openai.com/v1/chat/completions",
        "headers": {
          "Authorization": "Bearer ${credential.openAI_key}"
        },
        "request_body": "{ \"model\": \"${parameters.model}\", \"messages\": [{\"role\":\"developer\",\"content\":\"${parameters.system_prompt}\"},${parameters._chat_history:-}{\"role\":\"user\",\"content\":\"${parameters.user_prompt}\"}${parameters._interactions:-}], \"reasoning_effort\":\"low\"${parameters.tool_configs:-}}"
      }
    ]
  }
}
```

**Sample Response:**
```json
{
  "task_id": "ttwt9JwBjrSrr3-X8_b3",
  "status": "CREATED",
  "model_id": "t9wt9JwBjrSrr3-X9PYG"
}
```

Deploy the model:

```json
POST /_plugins/_ml/models/t9wt9JwBjrSrr3-X9PYG/_deploy
```

## Step 6: Create Agent with Docling MCP Tools

Create an agent that combines OpenSearch built-in tools with Docling MCP external tools:

```json
POST /_plugins/_ml/agents/_register
{
  "name": "Article Summarization Agent with Docling MCP",
  "type": "conversational",
  "description": "Agent that finds articles, processes PDFs with Docling MCP, and generates summaries",
  "llm": {
    "model_id": "<model_id_from_step_5>",
    "parameters": {
      "max_iteration": 20
    }
  },
  "memory": {
    "type": "conversation_index"
  },
  "parameters": {
    "_llm_interface": "openai/v1/chat/completions",
    "mcp_connectors": [
      {
        "mcp_connector_id": "<docling_mcp_connector_id>"
      }
    ]
  },
  "tools": [
    {
      "type": "ListIndexTool",
      "name": "ListIndexTool",
      "description": "List available indexes in OpenSearch"
    },
    {
      "type": "IndexMappingTool",
      "name": "IndexMappingTool",
      "description": "Get the mapping/schema of an index"
    },
    {
      "type": "SearchIndexTool",
      "name": "SearchIndexTool",
      "description": "Search documents in an index"
    },
    {
      "type": "QueryPlanningTool",
      "name": "QueryPlanningTool",
      "description": "Plan and generate OpenSearch DSL queries"
    }
  ],
  "app_type": "os_chat"
}
```

**Sample Response:**
```json
{
  "agent_id": "0Nwy9JwBjrSrr3-XDPaS"
}
```

The agent now has access to:

**Built-in OpenSearch Tools:**
- `ListIndexTool` - List available indexes
- `IndexMappingTool` - Get index schema
- `SearchIndexTool` - Search documents
- `QueryPlanningTool` - Generate DSL queries

**External Docling MCP Tools** (automatically available):
- `convert_document_into_docling_document` - Convert PDFs to structured format
- `export_docling_document_to_markdown` - Extract text content
- `get_overview_of_document_anchors` - Get document structure
- `search_for_text_in_document_anchors` - Search within documents
- `get_text_of_document_item_at_anchor` - Get specific text sections
- `is_document_in_local_cache` - Check if document is cached

## Step 7: Create Agentic Search Pipeline

Create a search pipeline that uses the agent:

```json
PUT _search/pipeline/article-summarization-pipeline
{
  "request_processors": [
    {
      "agentic_query_translator": {
        "agent_id": "<agent_id_from_step_6>"
      }
    }
  ],
  "response_processors": [
    {
      "agentic_context": {
        "agent_steps_summary": true,
        "dsl_query": true
      }
    }
  ]
}
```

## Step 8: Run Agentic Search Queries

Now you can ask natural language questions and let the agent orchestrate the tools!

### Example 1: Summarize Latest Document Processing Article

```json
POST articles-index/_search?search_pipeline=article-summarization-pipeline
{
  "query": {
    "agentic": {
      "query_text": "Give me a summary of the latest article from the document-processing category"
    }
  }
}
```

**Agent orchestration:**
1. `ListIndexTool` → Discovers `articles-index`
2. `IndexMappingTool` → Understands schema (category, publication_date, pdf_url fields)
3. `QueryPlanningTool` → Generates query to find latest article in "document-processing" category
4. `SearchIndexTool` → Executes search and retrieves article metadata
5. `convert_document_into_docling_document` (Docling MCP) → Fetches and converts PDF from URL
6. `export_docling_document_to_markdown` (Docling MCP) → Exports document to markdown
7. LLM → Generates summary from markdown content

**Expected Response:**
```json
{
  "took": 15234,
  "hits": {
    "total": {"value": 1},
    "hits": [
      {
        "_index": "articles-index",
        "_id": "doc1",
        "_source": {
          "title": "Docling Technical Report",
          "category": "document-processing",
          "publication_date": "2024-08-19",
          "pdf_url": "https://arxiv.org/pdf/2408.09869",
          "department": "Product Management"
        }
      }
    ]
  },
  "ext": {
    "agent_steps_summary": "I have these tools available: [ListIndexTool, IndexMappingTool, SearchIndexTool, QueryPlanningTool, convert_document_into_docling_document, export_docling_document_to_markdown, get_overview_of_document_anchors]\nFirst I used: ListIndexTool — found articles-index\nSecond I used: IndexMappingTool — understood schema with category and publication_date fields\nThird I used: QueryPlanningTool — generated query to find latest document in 'document-processing' category\nFourth I used: SearchIndexTool — found 'Docling Technical Report' published on 2024-08-19\nFifth I used: convert_document_into_docling_document — input: {\"source\": \"https://arxiv.org/pdf/2408.09869\"}; successfully converted PDF, document_key: abc123\nSixth I used: export_docling_document_to_markdown — input: {\"document_key\": \"abc123\"}; extracted full text in markdown format\nSeventh: Generated summary from the markdown content.\n\nSummary: Docling is a lightweight, MIT‑licensed open‑source package that converts PDF documents into structured formats. It leverages state‑of‑the‑art AI models—DocLayNet for layout analysis and TableFormer for table structure recognition—to achieve high accuracy while running efficiently on commodity hardware with minimal resources. The modular code interface makes it straightforward to extend the tool or integrate new models and features.",
    "memory_id": "2dyh85wBjrSrr3-XIvOC",
    "dsl_query": "{\"query\":{\"bool\":{\"filter\":[{\"term\":{\"category\":\"document-processing\"}}]}},\"sort\":[{\"publication_date\":{\"order\":\"desc\"}}],\"size\":1}",
  }
}
```

### Example 2: Summarize Latest Transformers Article

```json
POST articles-index/_search?search_pipeline=article-summarization-pipeline
{
  "query": {
    "agentic": {
      "query_text": "Summarize the most recent article about transformers"
    }
  }
}
```

**Agent orchestration:**
1. Searches for latest article in "transformers" category
2. Finds "BERT: Pre-training..." (2018-10-11)
3. Converts PDF with Docling MCP
4. Exports to markdown
5. Generates summary

**Expected Response:**
```json
{
  "ext": {
    "agent_steps_summary": "Found 'BERT: Pre-training of Deep Bidirectional Transformers' as the most recent transformers article. Converted the PDF and extracted the content.\n\nSummary: BERT introduces a new language representation model that uses bidirectional transformers for pre-training. Unlike previous models that use unidirectional context, BERT is designed to pre-train deep bidirectional representations by jointly conditioning on both left and right context in all layers. The paper demonstrates state-of-the-art results on eleven natural language processing tasks.",
    "dsl_query": "{\"query\":{\"bool\":{\"filter\":[{\"term\":{\"category\":\"transformers\"}}]}},\"sort\":[{\"publication_date\":{\"order\":\"desc\"}}],\"size\":1}"
  }
}
```

### Example 3: Find and Summarize Specific Article

```json
POST articles-index/_search?search_pipeline=article-summarization-pipeline
{
  "query": {
    "agentic": {
      "query_text": "Find the Attention Is All You Need paper and give me a brief summary"
    }
  }
}
```

**Agent orchestration:**
1. Searches for article with title matching "Attention Is All You Need"
2. Converts PDF with Docling MCP
3. Exports to markdown
4. Generates summary

**Expected Response:**
```json
{
  "ext": {
    "agent_steps_summary": "Found 'Attention Is All You Need' paper. Processed the PDF and generated summary.\n\nSummary: This seminal paper introduces the Transformer architecture, a novel neural network model based entirely on attention mechanisms, dispensing with recurrence and convolutions. The Transformer uses multi-head self-attention to process sequences in parallel, achieving superior results on machine translation tasks while being more parallelizable and requiring significantly less time to train than previous architectures.",
    "dsl_query": "{\"query\":{\"match\":{\"title\":\"Attention Is All You Need\"}}}"
  }
}
```

### Example 4: Latest Article Across All Categories

```json
POST articles-index/_search?search_pipeline=article-summarization-pipeline
{
  "query": {
    "agentic": {
      "query_text": "What is the most recent article in the repository? Give me a summary."
    }
  }
}
```

**Agent orchestration:**
1. Queries for article with latest publication_date (no category filter)
2. Finds "Docling Technical Report" (2024-08-19)
3. Processes with Docling MCP
4. Generates summary

## Understanding the Agent Workflow

When you ask: **"Give me a summary of the latest article from the document-processing category"**

The agent automatically orchestrates these steps:

1. **ListIndexTool** (OpenSearch) → Discovers the `articles-index`
2. **IndexMappingTool** (OpenSearch) → Understands the schema (category, publication_date, pdf_url fields)
3. **QueryPlanningTool** (OpenSearch) → Generates a query to find latest article in "document-processing" category
4. **SearchIndexTool** (OpenSearch) → Executes the search and retrieves the article metadata
5. **convert_document_into_docling_document** (Docling MCP) → Fetches and converts the PDF from the URL
6. **export_docling_document_to_markdown** (Docling MCP) → Exports the document to markdown text
7. **LLM** → Generates a summary from the markdown content

All of this happens automatically - you just ask the question in natural language!

## Key Benefits

✅ **No manual coding** - Just natural language queries via REST API
✅ **Automatic orchestration** - Agent decides which tools to use and in what order
✅ **Real-time processing** - PDFs are fetched and processed on-demand
✅ **Flexible** - Works with a local path or a remote URL
✅ **Transparent** - Agent steps are logged and visible
✅ **Copy-paste ready** - All commands work in OpenSearch Dashboard console

## Related Documentation

- [OpenSearch MCP Server Documentation](https://docs.opensearch.org/latest/vector-search/ai-search/agentic-search/mcp-server/)
- [Docling MCP Documentation](../../README.md)
- [OpenSearch Tools Index](https://docs.opensearch.org/latest/ml-commons-plugin/agents-tools/tools/index/)
- [MCP Protocol Specification](https://modelcontextprotocol.io/)
- [OpenSearch Documentation](https://opensearch.org/docs/latest/)