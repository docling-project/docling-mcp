# Examples on how to use Mellea

This example folder shows how you can leverage [Mellea](https://github.com/generative-computing/mellea) to build effective agents working on documents.

The examples here are meant for educational purposes and show a good balance between total freedom for the agent as well as strict control via programmatic pathways.

## Writing documents

In this example, we let the LLM write a report on a particular subject. Contrary to other applications where the LLM writes the entire document in 1 shot (eg in MarkDown), here we let the LLM first plan by building an outline and then expand on the outline.  

You can run the example with,

```
uv run python ./examples/mellea/example_01_write_report.py
```

## Editing documents

In this example, we let the LLM edit an already existing document. Contrary to othe applications, we do not provide the entire document in markdown/html in the context window, but rather provide the LLM first with a shortened outline that allows it to pinpoint the sections it needs to update. Only those sections are provided to the LLM to then edit, making for a shorter context window and less chance for error's. 

You can run the example with,

```
uv run python ./examples/mellea/example_02_edit_report.py
```
