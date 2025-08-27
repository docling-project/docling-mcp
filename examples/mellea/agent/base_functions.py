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
from mellea.stdlib.requirement import Requirement, simple_validate
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

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def find_json_dicts(text: str) -> list[dict]:
    """
    Extract JSON dictionaries from ```json code blocks
    """
    pattern = r"```json\s*(.*?)\s*```"
    matches = re.findall(pattern, text, re.DOTALL)

    calls = []
    for i, json_content in enumerate(matches):
        try:
            # print(f"call {i}: {json_content}")
            calls.append(json.loads(json_content))
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON in match {i}: {e}")

    return calls


def find_crefs(text: str) -> list[RefItem]:
    """
    Check if a string matches the pattern ```markdown(.*)?```
    """
    labels: str = "|".join([_ for _ in DocItemLabel])
    pattern = rf"#/({labels})/\d+"

    match = re.search(pattern, text, re.DOTALL)

    refs = []
    for i, _ in enumerate(match):
        refs.append(RefItem(cref=_.group(0)))

    return refs


def has_crefs(text: str) -> bool:
    return len(find_crefs) > 0


def create_document_outline(doc: DoclingDocument) -> str:
    label_counter: dict[DocItemLabel, int] = {
        DocItemLabel.TABLE: 0,
        DocItemLabel.PICTURE: 0,
        DocItemLabel.TEXT: 0,
    }

    lines = []
    for item, level in doc.iterate_items(with_groups=True):
        if isinstance(item, TitleItem):
            lines.append(f"title (reference={item.self_ref}): {item.text}")

        elif isinstance(item, SectionHeaderItem):
            lines.append(
                f"section-header (level={item.level}, reference={item.self_ref}): {item.text}"
            )

        elif isinstance(item, ListItem):
            continue

        elif isinstance(item, TextItem):
            lines.append(f"{item.label} (reference={item.self_ref})")

        elif isinstance(item, TableItem):
            label_counter[item.label] += 1
            lines.append(
                f"{item.label} {label_counter[item.label]} (reference={item.self_ref})"
            )

        elif isinstance(item, PictureItem):
            label_counter[item.label] += 1
            lines.append(
                f"{item.label} {label_counter[item.label]} (reference={item.self_ref})"
            )

    outline = "\n\n".join(lines)

    return outline


def find_outline(text: str) -> DoclingDocument | None:
    starts = ["paragraph", "list", "table", "figure", "picture"]

    md = find_markdown_code_block(text)

    if md:
        converter = DocumentConverter(allowed_formats=[InputFormat.MD])

        buff = BytesIO(md.encode("utf-8"))
        doc_stream = DocumentStream(name="tmp.md", stream=buff)

        conv: ConversionResult = converter.convert(doc_stream)

        lines = []
        for item, level in conv.document.iterate_items(with_groups=True):
            if isinstance(item, TitleItem) or isinstance(item, SectionHeaderItem):
                continue
            elif isinstance(item, TextItem):
                pattern = rf"^({'|'.join(starts)}):\s(.*)\.$"
                match = bool(re.match(pattern, text, re.DOTALL))
                if match is None:
                    lines.append(item.text)
            else:
                continue

        if len(lines) > 0:
            message = f"Every content line should start with one out of the following choices: {starts}. The following lines need to be updated: {'\n'.join(lines)}"
            logger.error(message)

            return None
        else:
            return conv.document
    else:
        return None


def validate_outline_format(text: str) -> bool:
    logger.info(f"testing validate_outline_format for {text[0:64]}")
    return find_outline(text) is not None


def serialize_item_to_markdown(item: TextItem, doc: DoclingDocument) -> str:
    """Serialize a text item to markdown format using existing serializer."""
    from docling_core.transforms.serializer.markdown import (
        MarkdownDocSerializer,
        MarkdownParams,
    )

    serializer = MarkdownDocSerializer(doc=doc, params=MarkdownParams())

    result = serializer.serialize(item=item)
    return result.text


def serialize_table_to_html(table: TableItem, doc: DoclingDocument) -> str:
    from docling_core.transforms.serializer.html import (
        HTMLTableSerializer,
        HTMLDocSerializer,
    )

    # Create the table serializer
    table_serializer = HTMLTableSerializer()

    # Create a document serializer (needed as dependency)
    doc_serializer = HTMLDocSerializer(doc=doc)

    # Serialize the table
    result = table_serializer.serialize(
        item=table, doc_serializer=doc_serializer, doc=doc
    )

    return result.text


def find_html_code_block(text: str) -> str | None:
    """
    Check if a string matches the pattern ```html(.*)?```
    """
    pattern = r"```html(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1) if match else None


def has_html_code_block(text: str) -> bool:
    """
    Check if a string contains a html code block pattern anywhere in the text
    """
    logger.info(f"testing has_html_code_block for {text[0:64]}")
    return find_html_code_block(text) is not None


def find_markdown_code_block(text: str) -> str | None:
    """
    Check if a string matches the pattern ```(md|markdown)(.*)?```
    """
    pattern = r"```(md|markdown)(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(2) if match else None


def has_markdown_code_block(text: str) -> bool:
    """
    Check if a string contains a markdown code block pattern anywhere in the text
    """
    logger.info(f"testing has_markdown_code_block for {text[0:64]}")
    return find_markdown_code_block(text) is not None


def convert_html_to_docling_table(text: str) -> list[TableItem] | None:
    text_ = find_html_code_block(text)
    if text_ is None:
        text_ = text  # assume the entire text is html

    try:
        converter = DocumentConverter(allowed_formats=[InputFormat.HTML])

        buff = BytesIO(text.encode("utf-8"))
        doc_stream = DocumentStream(name="tmp.html", stream=buff)

        conv: ConversionResult = converter.convert(doc_stream)

        if conv.status == ConversionStatus.SUCCESS:
            return conv.document.tables

    except Exception as exc:
        logger.error(exc)
        return None

    return None


def validate_html_to_docling_table(text: str) -> bool:
    logger.info(f"validate_html_to_docling_table for {text[0:64]}")
    return convert_html_to_docling_table is not None


def convert_markdown_to_docling_document(text: str) -> DoclingDocument | None:
    text_ = find_markdown_code_block(text)
    if text_ is None:
        text_ = text  # assume the entire text is html

    try:
        converter = DocumentConverter(allowed_formats=[InputFormat.MD])

        buff = BytesIO(text_.encode("utf-8"))
        doc_stream = DocumentStream(name="tmp.md", stream=buff)

        conv: ConversionResult = converter.convert(doc_stream)

        if conv.status == ConversionStatus.SUCCESS:
            return conv.document
    except Exception as exc:
        return None

    return None


def validate_markdown_to_docling_document(text: str) -> bool:
    logger.info(f"testing validate_markdown_docling_document for {text[0:64]}")
    return convert_markdown_to_docling_document(text) is not None


def convert_html_to_docling_document(text: str) -> DoclingDocument | None:
    text_ = find_html_code_block(text)
    if text_ is None:
        text_ = text  # assume the entire text is html

    try:
        converter = DocumentConverter(allowed_formats=[InputFormat.HTML])

        buff = BytesIO(text.encode("utf-8"))
        doc_stream = DocumentStream(name="tmp.html", stream=buff)

        conv: ConversionResult = converter.convert(doc_stream)

        if conv.status == ConversionStatus.SUCCESS:
            return conv.document
    except Exception as exc:
        logger.error(f"error: {exc}")
        return None

    return None


def validate_html_to_docling_document(text: str) -> bool:
    logger.info(f"testing validate_html_docling_document for {text[0:64]}")
    return convert_html_to_docling_document(text) is not None


def insert_document(
    item: NodeItem, doc: DoclingDocument, updated_doc: DoclingDocument
) -> DoclingDocument:
    group_item = GroupItem(
        label=GroupLabel.UNSPECIFIED,
        name="inserted-group",
        self_ref="#",  # temporary placeholder
    )

    if isinstance(item, ListItem):
        # we should delete all the children of the list-item and put the text to ""
        raise ValueError("ListItem insertion is not yet supported!")

    doc.replace_item(old_item=item, new_item=group_item)

    to_item: dict[str, NodeItem] = {}

    for item, level in updated_doc.iterate_items(with_groups=True):
        if isinstance(item, GroupItem) and item.self_ref == "#/body":
            to_item[item.self_ref] = group_item

        elif item.parent is None:
            logger.error(f"Item with null parent: {item}")

        elif item.parent.cref not in to_item:
            logger.error(f"Item with unknown parent: {item}")

        elif isinstance(item, GroupItem):
            g = doc.add_group(
                name=item.name,
                label=item.label,
                parent=to_item[item.parent.cref],
            )

        elif isinstance(item, ListItem):
            li = doc.add_list_item(
                text=item.text,
                formatting=item.formatting,
                parent=to_item[item.parent.cref],
            )
            to_item[item.self_ref] = li

        elif isinstance(item, TextItem):
            te = doc.add_text(
                text=item.text,
                label=item.label,
                formatting=item.formatting,
                parent=to_item[item.parent.cref],
            )
            to_item[item.self_ref] = te

        elif isinstance(item, TableItem):
            if len(item.captions) > 0:
                caption = doc.add_text(
                    label=DocItemLabel.CAPTION, text=item.captions[0].text
                )
                te = doc.add_table(
                    data=item.data,
                    caption=caption,
                )
                to_item[item.self_ref] = te
            else:
                te = doc.add_table(
                    data=item.data,
                )
                to_item[item.self_ref] = te

        else:
            logger.warning(f"No support to insert items of label: {item.label}")
