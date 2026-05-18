import copy
import logging
import re
from datetime import datetime
from enum import Enum
from io import BytesIO
from typing import ClassVar
import json

from pydantic import BaseModel, Field, validator

from smolagents import MCPClient, Tool, ToolCollection
from smolagents.models import ChatMessage, MessageRole, Model

from mellea.backends import model_ids
from mellea.backends.model_ids import ModelIdentifier
from mellea.stdlib.requirements import Requirement, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.document import ConversionResult
from docling.document_converter import DocumentConverter
from docling_core.types.doc.document import (
    ContentLayer,
    DocItemLabel,
    DoclingDocument,
    NodeItem,
    GroupItem,
    GroupLabel,
    DocItem,
    LevelNumber,
    ListItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
    TitleItem,
    RefItem,
    PictureItem,
)
from docling_core.types.io import DocumentStream

from examples.mellea.agent_models import setup_local_session

# from examples.smolagents.agent_tools import MCPConfig, setup_mcp_tools
from examples.mellea.resources.prompts import (
    SYSTEM_PROMPT_FOR_TASK_ANALYSIS,
    SYSTEM_PROMPT_FOR_OUTLINE,
    SYSTEM_PROMPT_FOR_EDITING_DOCUMENT,
    SYSTEM_PROMPT_FOR_EDITING_TABLE,
    SYSTEM_PROMPT_EXPERT_WRITER,
    SYSTEM_PROMPT_EXPERT_TABLE_WRITER,
)
from abc import abstractmethod

from examples.mellea.agent.base import DoclingAgentType, BaseDoclingAgent

from examples.mellea.agent.base_functions import (
    find_json_dicts,
    find_crefs,
    has_crefs,
    create_document_outline,
    serialize_item_to_markdown,
    serialize_table_to_html,
    find_html_code_block,
    has_html_code_block,
    find_markdown_code_block,
    has_markdown_code_block,
    convert_html_to_docling_table,
    validate_html_to_docling_table,
    convert_markdown_to_docling_document,
    validate_markdown_to_docling_document,
    insert_document,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DoclingEditingAgent(BaseDoclingAgent):
    system_prompt_for_editing_document: ClassVar[str] = (
        SYSTEM_PROMPT_FOR_EDITING_DOCUMENT
    )
    system_prompt_for_editing_table: ClassVar[str] = SYSTEM_PROMPT_FOR_EDITING_TABLE

    system_prompt_expert_writer: ClassVar[str] = SYSTEM_PROMPT_EXPERT_WRITER

    def __init__(self, *, model_id: ModelIdentifier, tools: list[Tool]):
        super().__init__(
            agent_type=DoclingAgentType.DOCLING_DOCUMENT_EDITOR,
            model_id=model_id,
            tools=tools,
        )

    def run(self, task: str, document: DoclingDocument, **kwargs) -> DoclingDocument:
        op = self._identify_document_items(task=task, document=document)

        if op["operation"] == "update_content":
            self._update_content(task=task, document=document, sref=op["ref"])
        elif op["operation"] == "rewrite_content":
            self._rewrite_content(
                task=task,
                document=document,
                refs=op["refs"],
            )
        elif op["operation"] == "delete_content":
            self._delete_content(task=task, document=document, refs=op["refs"])
        elif op["operation"] == "update_section_heading_level":
            self._update_section_heading_level(
                task=task, document=document, to_level=op["to_level"]
            )
        else:
            message = f"Could not execute operate op: {op}"
            logger.info(message)
            raise ValueError(message)

    def _identify_document_items(
        self,
        task: str,
        document: DoclingDocument,
        loop_budget: int = 5,
    ) -> list[RefItem]:
        logger.info(f"task: {task}")

        outline = create_document_outline(doc=document)

        context = rf"""Given the current outline of the document:
```
{outline}
```

"""

        identification = rf"""To accomplish the following task:

{task}

We first need to:
    - identify from the outline all the document items that are relevant
    - plan the operations needed to update the document

Now, provide me the operations (encapsulated in on ore more ```json...```) and their references to execute the task!"""

        prompt = f"{context}{identification}"

        m = setup_local_session(
            model_id=self.model_id,
            system_prompt=self.system_prompt_for_editing_document,
        )

        answer = m.instruct(
            prompt,
            strategy=RejectionSamplingStrategy(loop_budget=loop_budget),
        )

        ops = find_json_dicts(text=answer.value)

        if len(ops) == 0:
            raise ValueError(f"No operation is detected")

        if "operation" not in ops[0]:
            raise ValueError(f"`operation` not in op: {ops[0]}")

        return ops[0]

    def _update_content(self, task: str, document: DoclingDocument, sref: str):
        logger.info("_update_content_of_document_items")

        ref = RefItem(cref=sref)
        item = ref.resolve(document)

        if isinstance(item, TableItem):
            self._update_content_of_table(task=task, document=document, table=item)

        elif isinstance(item, TextItem):
            self._update_content_of_textitem(task=task, document=document, item=item)

        else:
            logger.warning(
                f"Dont know how to update the item (of label={item.label}) for task: {task}"
            )

    def _update_content_of_table(
        self,
        task: str,
        document: DoclingDocument,
        table: TableItem,
        loop_budget: int = 5,
    ):
        logger.info("_update_content_of_table")

        html_table = serialize_table_to_html(table=table, doc=document)

        prompt = f"""Given the following HTML table,

```html
{html_table}
```

Execute the following task: {task}
"""
        # logger.info(f"prompt: {prompt}")

        m = setup_local_session(
            model_id=self.model_id,
            system_prompt=self.system_prompt_for_editing_table,
        )

        answer = m.instruct(
            prompt,
            strategy=RejectionSamplingStrategy(loop_budget=loop_budget),
            requirements=[
                Requirement(
                    description="Put the resulting HTML table in the format ```html <insert-content>```",
                    validation_fn=simple_validate(has_html_code_block),
                ),
                Requirement(
                    description="The HTML table should have a valid formatting.",
                    validation_fn=simple_validate(validate_html_to_docling_table),
                ),
            ],
        )

        logger.info(f"response: {answer.value}")

        new_tables = convert_html_to_docling_table(text=answer.value)

        if new_tables and len(new_tables) == 1:
            table.data = new_tables[0].data
        elif new_tables and len(new_tables) > 1:
            logger.error("too many tables returned ...")
            table.data = new_tables[0].data

    def _update_content_of_textitem(
        self,
        task: str,
        document: DoclingDocument,
        item: TextItem,
        loop_budget: int = 5,
    ):
        logger.info("_update_content_of_text")

        text = serialize_item_to_markdown(item=item, doc=document)

        prompt = f"""Given the following {item.label},

```md
{text}
```

Execute the following task: {task}
"""
        # logger.info(f"prompt: {prompt}")

        m = setup_local_session(
            model_id=self.model_id,
            system_prompt=self.system_prompt_for_editing_table,
        )

        answer = m.instruct(
            prompt,
            strategy=RejectionSamplingStrategy(loop_budget=loop_budget),
        )
        # logger.info(f"response: {answer.value}")

        updated_doc = convert_markdown_to_docling_document(text=answer.value)

        document = insert_document(item=item, doc=document, updated_doc=updated_doc)

    def _delete_content_of_document_items(
        self, task: str, document: DoclingDocument, refs: list[RefItem]
    ):
        logger.info("_delete_content_of_document_items")

    def _append_content_of_document_items(
        self,
        task: str,
        document: DoclingDocument,
        ref: RefItem,
        labels: list[DocItemLabel],
    ):
        logger.info("_append_content_of_document_items")

    def _update_section_heading_level(
        self, task: str, document: DoclingDocument, to_level: dict[str, int]
    ):
        for sref, level in to_level.items():
            ref = RefItem(cref=sref)
            item = ref.resolve(document)

            if isinstance(item, SectionHeaderItem):
                item.level = level
            else:
                logger.warning(f"{sref} is not SectionHeaderItem {item.label}")

    def _rewrite_content(self, task: str, document: DoclingDocument, refs: list[str]):
        logger.info("_update_content_of_text")

        texts = []
        for sref in refs:
            ref = RefItem(cref=sref)
            item = ref.resolve(document)

            texts.append(serialize_item_to_markdown(item=item, doc=document))

        text = "\n\n".join(texts)

        prompt = f"""Given the following text-section in markdown,

```md
{text}
```

Execute the following task: {task}
"""
        logger.info(f"prompt: {prompt}")

        m = setup_local_session(
            model_id=self.model_id,
            system_prompt=self.system_prompt_for_expert_writer,
        )

        answer = m.instruct(
            prompt,
            strategy=RejectionSamplingStrategy(loop_budget=loop_budget),
        )
        logger.info(f"response: {answer.value}")

        updated_doc = convert_markdown_to_docling_document(text=answer.value)

        ref = RefItem(cref=refs[0])
        item = ref.resolve(document)

        document = insert_document(item=item, doc=document, updated_doc=updated_doc)
