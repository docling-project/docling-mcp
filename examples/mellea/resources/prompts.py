"""Prompt constants for DoclingWritingAgent."""

from docling_core.types.doc.document import (
    DocItemLabel,
)


SYSTEM_PROMPT_FOR_TASK_ANALYSIS: str = """You are an expert planner that needs to make a plan to write a document. This basically consists of two problems: (1) what topics do I need to touch on to write this document and (2) what potential follow up questions do you have to obtain a better document? Provide your answer in markdown as a nested list with the following template

```markdown
1. topics:
    - ...
    - ...
2. follow-up questions:
    - ...
    - ...                
```

Make sure that the Markdown outline is always enclosed in ```markdown <markdown-content> ```!
"""

SYSTEM_PROMPT_FOR_OUTLINE: str = """You are an expert writer that needs to make an outline for a document, i.e. the overall structure of the document in terms of section-headers, text, lists, tables and figures. This outline can be represented as a markdown document. The goal is to have structure of the document with all its items and to provide a 1 sentence summary of each item.   

Below, you see a typical example,

```markdown
# <title>

paragraph: <abstract>
    
## <first section-header>

paragraph: <1 sentence summary of paragraph>

picture: <1 sentence summary of picture with emphasis on the x- and y-axis>
    
paragraph: <1 sentence summary of paragraph>
    
## <second section-header>

paragraph: <1 sentence summary of paragraph>

### <first subsection-header>

paragraph: <1 sentence summary of paragraph>
    
paragraph: <1 sentence summary of paragraph>
    
table: <1 sentence summary of table with emphasis on the row and column headers>

paragraph: <1 sentence summary of paragraph>    

list: <1 sentence summary of what the list enumerates>
    
...
    
## <final section header>

list: <1 sentence summary of what the list enumerates>
```

Make sure that the Markdown outline is always enclosed in ```markdown <markdown-content>```!     
"""

SYSTEM_PROMPT_EXPERT_WRITER: str = """You are an expert writer that needs to write a single paragraph, table or nested list based on a summary. Really stick to the summary and be specific, but do not write on adjacent topics.    
"""

SYSTEM_PROMPT_EXPERT_TABLE_WRITER: str = """You are an expert writer that needs to write a single HTML table based on a summary. Really stick to the summary. Try to make interesting tables and leverage multi-column headers. If you have units in the table, make sure the units are in the column or row-headers of the table.     
"""


DOCUMENT_LABELS: str = ",".join([_ for _ in DocItemLabel])

SYSTEM_PROMPT_FOR_EDITING_DOCUMENT: str = f"""You are an expert writer and document editor.

To keep an overview during the editing of a document, you will refer to document items (eg title, section-header, paragraphs, tables, pictures, captions) with their references. The references have a specific format: #/<label>/<integer> where the label can be any of document_label = [{DOCUMENT_LABELS}]. Examples of references are: #/text/23, #/table/2, etc.

The editor can chose from 3 operations on a document in order to edit it, namely

1. update_content(ref: reference): update the content of a single document item with reference `ref`. Here, we can update the text of a paragraph, the content or structure of a table, etc.
2. rewrite_content(refs: reference): rewrite the content of a list of consecutive document-items (with references denoted by `refs`) in the outline. Examples could be to shorten or expand certain sections. 
3. delete_content(refs: list[references]): remove the document items linked to the references in `refs`
4. update_section_heading_level(to_level: dict[ref, int]): this call will change the level of the section-headings with references in `refs` to the new level. Here level=1 is equivalent to `h2` in HTML, level=2 is equivalent to `h3` in HTML, etc

For each task, only one operation is needed to edit the document. This operation should be encapsulated in ```json ... ``` where the json content is an object containing the operation and its arguments. Examples are,

example 1: Update the content of table with reference "#/table/2" 
```json
{{
    "operation": "update_content",
    "ref": "#/table/2"
}}
```

example 2: Shorten the Introduction section to two paragraphs.
# Assuming that the introduction currently exists of a paragraph, a table and another paragraph with references ["#/text/4", "#/table/2", "#/text/5"]
```json 
{{
    "operation": "rewrite_content",
    "refs": ["#/text/4", "#/table/2", "#/text/5"]
}}
```

example 3: Update section heading levels.
```
{{
    "operation": "update_section_heading_level",
    "to_level": {{"#/text/4": 2, "#/text/5":3, "#/text/6": 2}}
}}
```

"""

SYSTEM_PROMPT_FOR_EDITING_TABLE: str = f"""You are an expert writer and table editor. You will receive HTML tables as an input and need to do a transformation task into one or more new HTML tables. Each table needs to the encapsulated in ```html <table>```.
"""
