### Llama Index with Milvus RAG

Install Docling MCP with the `[llama-index-rag]` extra:

```sh
pip install "docling[llama-index-rag]"
```

Copy the .env.example file to .env in the root of the project.

```sh
cp .env.example .env
```

If you want to use the RAG Milvus functionality edit the new .env file to set both environment variables.

```text
RAG_ENABLED=true
OLLAMA_MODEL=granite3.2:latest
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
```

Note:

ollama can be downloaded here https://ollama.com/. Once you have ollama download the model you want to use and then add the model string to the .env file.

For example we are using `granite3.2:latest` to perform the RAG search.

To download this model run:

```sh
ollama pull granite3.2:latest
```

When using the docling-mcp server with RAG this would be a simple example prompt:

```prompt
Process this file /Users/name/example/mock.pdf 

Upload it to the vector store. 

Then summarize xyz that is contained within the document.
```

Known issues

When restarting the MCP client (e.g. Claude desktop) the client sometimes errors due to the `.milvus_demo.db.lock` file. Delete this before restarting.
