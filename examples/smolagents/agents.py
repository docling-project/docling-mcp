import copy
import logging
import re
from datetime import datetime
from enum import Enum
from io import BytesIO
from typing import ClassVar

from pydantic import BaseModel, Field, validator

from smolagents import MCPClient, Tool, ToolCollection
from smolagents.models import ChatMessage, MessageRole, Model

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.document import ConversionResult
from docling.document_converter import DocumentConverter
from docling_core.types.doc.document import (
    ContentLayer,
    DocItemLabel,
    DoclingDocument,
    GroupItem,
    LevelNumber,
    ListItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
    TitleItem,
)
from docling_core.types.io import DocumentStream

from examples.smolagents.agent_model import ModelConfig, setup_local_model
from examples.smolagents.agent_tools import MCPConfig, setup_mcp_tools
from examples.smolagents.resources.prompts import (
    SYSTEM_PROMPT_FOR_TASK_ANALYSIS,
    SYSTEM_PROMPT_FOR_OUTLINE,
    SYSTEM_PROMPT_EXPERT_WRITER,
    SYSTEM_PROMPT_EXPERT_TABLE_WRITER,
)
from abc import abstractmethod


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DoclingAgentType(Enum):
    """Enumeration of supported agent types."""

    # Core agent types
    DOCLING_DOCUMENT_WRITER = "writer"

    def __str__(self) -> str:
        """Return the string value of the enum."""
        return self.value

    @classmethod
    def from_string(cls, value: str) -> "AgentType":
        """Create AgentType from string value."""
        for agent_type in cls:
            if agent_type.value == value:
                return agent_type
        raise ValueError(
            f"Invalid agent type: {value}. Valid types: {[t.value for t in cls]}"
        )

    @classmethod
    def get_all_types(cls) -> list[str]:
        """Get all available agent type strings."""
        return [agent_type.value for agent_type in cls]


class BaseDoclingAgent(BaseModel):
    agent_type: DoclingAgentType
    model: Model
    tools: list[Tool]
    chat_history: list[ChatMessage]
    max_iteration: int = 16

    class Config:
        arbitrary_types_allowed = True  # Needed for complex types like Model

    @abstractmethod
    def run(self, task: str, **kwargs) -> str:
        return


class DoclingWritingAgent(BaseDoclingAgent):
    task_analysis: DoclingDocument = DoclingDocument(name=f"report")

    system_prompt_for_task_analysis: ClassVar[str] = SYSTEM_PROMPT_FOR_TASK_ANALYSIS

    system_prompt_for_outline: ClassVar[str] = SYSTEM_PROMPT_FOR_OUTLINE

    system_prompt_expert_writer: ClassVar[str] = SYSTEM_PROMPT_EXPERT_WRITER

    system_prompt_expert_table_writer: ClassVar[str] = SYSTEM_PROMPT_EXPERT_TABLE_WRITER

    def __init__(self, *, model: Model, tools: list[Tool]):
        super().__init__(
            agent_type=DoclingAgentType.DOCLING_DOCUMENT_WRITER,
            model=model,
            tools=tools,
            chat_history=[],
        )

    def _init_chat_messages(
        self, *, system_prompt: str, user_prompt: str
    ) -> list[ChatMessage]:
        return [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=[{"type": "text", "text": system_prompt}],
            ),
            ChatMessage(
                role=MessageRole.USER,
                content=[{"type": "text", "text": user_prompt}],
            ),
        ]

    def run(self, task: str, **kwargs):
        # self._analyse_task_for_topics_and_followup_questions(task=task)

        # self._analyse_task_for_final_destination(task=task)

        document: DoclingDocument = self._make_outline_for_writing(task=task)

        document = self._populate_document_with_content(task=task, outline=document)

        fname = datetime.now().strftime("%Y%m%d_%H%M%S")

        document.save_as_markdown(filename=f"./scratch/{fname}.md", text_width=72)
        document.save_as_html(filename=f"./scratch/{fname}.html")
        document.save_as_json(filename=f"./scratch/{fname}.json")

        logger.info(f"report written to `./scratch/{fname}.json`")

    def _analyse_task_for_topics_and_followup_questions(self, *, task: str):
        chat_messages = self._init_chat_messages(
            system_prompt=self.system_prompt_for_task_analysis,
            user_prompt=f"{task}",
        )

        output = self.model.generate(messages=chat_messages)

        self.chat_history.extend(chat_messages)
        self.chat_history.append(output)

        results = self._analyse_output_into_docling_document(message=output)
        assert len(results) == 1, (
            "We only want to see a single response from the initial task analysis"
        )

        self.task_analysis = results[0]

        in_topics: bool = False
        in_questions: bool = False

        for item, level in self.task_analysis.iterate_items():
            if isinstance(item, ListItem) and item.text == "topics:":
                in_topics = True
            elif isinstance(item, ListItem) and item.text == "follow-up questions:":
                in_questions = True

    def _analyse_task_for_final_destination(self, *, task: str):
        return

    def _make_outline_for_writing(self, *, task: str) -> DoclingDocument:
        chat_messages = self._init_chat_messages(
            system_prompt=self.system_prompt_for_outline,
            user_prompt=f"{task}",
        )

        iteration = 0
        while iteration < self.max_iteration:
            iteration += 1
            logger.info(f"_make_outline_for_writing: iteration {iteration}")

            output = self.model.generate(messages=chat_messages)

            results = self._analyse_output_into_docling_document(message=output)

            if len(results) == 0:
                chat_messages.append(
                    ChatMessage(
                        role=MessageRole.USER,
                        content=[
                            {
                                "type": "text",
                                "text": f"I see now markdown section. Please try again and add a markdown section in the format ```markdown <insert-content>``` for task: {task}!",
                            }
                        ],
                    )
                )
                continue
            elif len(results) > 1:
                chat_messages.append(
                    ChatMessage(
                        role=MessageRole.USER,
                        content=[
                            {
                                "type": "text",
                                "text": f"I see multiple markdown sections. Please try again and only add a single markdown section in the format ```markdown <insert-content>``` for task: {task}!",
                            }
                        ],
                    )
                )
                continue
            else:
                logger.info("We obtained a markdown for the outline!")

            document = results[0]
            logger.info(f"outline: {document.export_to_markdown()}")

            starts = [
                "paragraph: ",
                "table: ",
                "picture: ",
                "list: ",
            ]
            lines = []
            for item, level in document.iterate_items(with_groups=True):
                if isinstance(item, TitleItem) or isinstance(item, SectionHeaderItem):
                    continue
                elif isinstance(item, TextItem):
                    good: bool = False
                    for start in starts:
                        if item.text.startswith(start):
                            good = True
                            break

                    if not good:
                        lines.append(item.text)

            logger.info(f"broken lines: {'\n'.join(lines)}")

            if len(lines) > 0:
                message = f"Every content line should start with one out of the following choices: {starts}. The following lines need to be updated: {'\n'.join(lines)}"
                chat_messages.append(
                    ChatMessage(
                        role=MessageRole.USER,
                        content=[{"type": "text", "text": message}],
                    )
                )
            else:
                self.chat_history.extend(chat_messages)
                self.chat_history.append(output)

                logger.info(
                    f"Finished an outline for document: {document.export_to_markdown()}"
                )
                return document

        raise ValueError("Could not make a correct outline!")

    def _update_document_with_content(
        self, *, document: DoclingDocument, content: DoclingDocument
    ) -> DoclingDocument:
        to_item: dict[str, NodeItem] = {}

        """
        for item, level in content.iterate_items(with_groups=True):
            print(
                "\t" * level,
                item.self_ref,
                f"({item.label}): ",
                item.parent,
            )
        """

        for item, level in content.iterate_items(with_groups=True):
            # print("\t"*level, item)
            # print("\t"*level, item.self_ref, f"({item.label}): ", item.parent)

            if isinstance(item, GroupItem) and item.self_ref == "#/body":
                to_item[item.self_ref] = document.body
            elif isinstance(item, GroupItem):
                if item.parent and item.parent.cref in to_item:
                    g = document.add_group(
                        name=item.name,
                        label=item.label,
                        parent=to_item[item.parent.cref],
                    )
                    to_item[item.self_ref] = g
                else:
                    print("adding: ", item)
                    g = document.add_group(
                        name=item.name, label=item.label, parent=None
                    )
                    to_item[item.self_ref] = g

            elif isinstance(item, ListItem):
                if item.parent and item.parent.cref in to_item:
                    li = document.add_list_item(
                        text=item.text,
                        formatting=item.formatting,
                        parent=to_item[item.parent.cref],
                    )
                    to_item[item.self_ref] = li
                else:
                    print("skipping: ", item)

            elif isinstance(item, TextItem):
                if item.parent and item.parent.cref in to_item:
                    te = document.add_text(
                        text=item.text,
                        label=item.label,
                        formatting=item.formatting,
                        parent=to_item[item.parent.cref],
                    )
                    to_item[item.self_ref] = te
                else:
                    print("skipping: ", item)
            else:
                print("skipping: ", item)

        """
        for item, level in document.iterate_items(with_groups=True):
            print(
                "\t" * level,
                item.self_ref,
                f"({item.label}): ",
                item.parent,
            )
        """

        return document

    def _populate_document_with_content(
        self, *, task: str, outline: DoclingDocument
    ) -> DoclingDocument:
        def update_headers(
            *, item: SectionHeaderItem, headers: dict[int, str]
        ) -> dict[int, str]:
            keys = copy.deepcopy(list(headers.keys()))
            for key in keys:
                if key > item.level:
                    del headers[key]

            headers[item.level] = item.text
            return headers

        headers: dict[int, str] = {}

        document = DoclingDocument(name=f"report on task: `{task}`")

        for item, level in outline.iterate_items(with_groups=True):
            if isinstance(item, TitleItem):
                headers[0] = item.text
                document.add_title(text=item.text)

            elif isinstance(item, SectionHeaderItem):
                logger.info(f"starting in {item.text}")
                headers = update_headers(item=item, headers=headers)
                document.add_heading(text=item.text, level=item.level)

            elif isinstance(item, TextItem):
                if item.text.startswith("paragraph:"):
                    summary = item.text.replace("paragraph: ", "").strip()

                    logger.info(f"need to write a paragraph: {summary})")
                    content = self._write_paragraph(
                        summary=summary, item_type="paragraph"
                    )
                    document.add_text(label=DocItemLabel.TEXT, text=content)

                elif item.text.startswith("list:"):
                    summary = item.text.replace("list:", "").strip()
                    logger.info(f"need to write a list: {summary}")

                    new_content = self._write_list(summary=summary)
                    document = self._update_document_with_content(
                        document=document, content=new_content
                    )

                elif item.text.startswith("table:"):
                    summary = item.text.replace("table:", "").strip()
                    logger.info(f"need to write a table: {summary}")

                    table_item = self._write_table(summary=summary)

                    caption = document.add_text(
                        label=DocItemLabel.CAPTION, text=summary
                    )
                    document.add_table(data=table_item.data, caption=caption)

        return document

    def _analyse_output_into_docling_document(
        self, message: ChatMessage, language: str = "markdown"
    ) -> list[DoclingDocument]:
        def extract_code_blocks(text, language: str):
            pattern = rf"```{language}\s*(.*?)\s*```"
            matches = re.findall(pattern, text, re.DOTALL)
            return matches

        converter = DocumentConverter(allowed_formats=[InputFormat.MD])

        result = []
        for mtch in extract_code_blocks(message.content, language=language):
            md_doc: str = mtch

            buff = BytesIO(md_doc.encode("utf-8"))
            doc_stream = DocumentStream(name="tmp.md", stream=buff)

            conv_result: ConversionResult = converter.convert(doc_stream)
            result.append(conv_result.document)

        return result

    def _write_paragraph(
        self, summary: str, item_type: str, task: str = "", hierarchy: list[str] = []
    ) -> str:
        chat_messages = self._init_chat_messages(
            system_prompt=self.system_prompt_expert_writer,
            user_prompt=(
                f"write me a single {item_type} that expands the following summary: {summary}"
            ),
        )
        output = self.model.generate(messages=chat_messages)
        return output.content

    def _write_list(
        self, summary: str, task: str = "", hierarchy: list[str] = []
    ) -> DoclingDocument | None:
        converter = DocumentConverter(allowed_formats=[InputFormat.MD])

        chat_messages = self._init_chat_messages(
            system_prompt=self.system_prompt_expert_writer,
            user_prompt=(
                f"write me a list (it can be nested) in markdown that expands the following summary: {summary}"
            ),
        )

        output = self.model.generate(messages=chat_messages)
        # print(output.content)

        md_doc: str = output.content  # extract_code_blocks(output.content)
        # print("md-doc:\n\n", md_doc)

        try:
            buff = BytesIO(md_doc.encode("utf-8"))
            doc_stream = DocumentStream(name="tmp.md", stream=buff)

            conv_result: ConversionResult = converter.convert(doc_stream)
            doc = conv_result.document

            if doc:
                return doc

        except Exception as exc:
            logger.error(f"error with html conversion: {exc}")

        return None

    def _write_table(self, summary: str, hierarchy: list[str] = []) -> TableItem | None:
        def extract_code_blocks(text):
            pattern = rf"```html(.*?)```"
            matches = re.findall(pattern, text, re.DOTALL)
            if len(matches) > 0:
                return matches[0]

            pattern = rf"<html>(.*?)</html>"
            matches = re.findall(pattern, text, re.DOTALL)
            if len(matches) > 0:
                return matches[0]

            return None

        chat_messages = self._init_chat_messages(
            system_prompt=self.system_prompt_expert_table_writer,
            user_prompt=(
                f"write me a single table in HTML that expands the following summary: {summary}"
            ),
        )

        doc = None

        converter = DocumentConverter(allowed_formats=[InputFormat.HTML])

        iteration = 0
        while iteration < self.max_iteration:
            iteration += 1

            output = self.model.generate(messages=chat_messages)
            # print("output:\n\n", output.content)

            html_doc: str = extract_code_blocks(output.content)
            # print("html-doc:\n\n", html_doc)

            try:
                buff = BytesIO(html_doc.encode("utf-8"))
                doc_stream = DocumentStream(name="tmp.html", stream=buff)

                conv_result: ConversionResult = converter.convert(doc_stream)
                doc = conv_result.document

                if doc:
                    for item, level in doc.iterate_items(with_groups=True):
                        if isinstance(item, TableItem):
                            return item

            except Exception as exc:
                logger.error(f"error with html conversion: {exc}")

        return None


def main():
    """
    model_config = ModelConfig(
        type="ollama",
        model_id="ollama/smollm2",  # , device="cpu", torch_dtype="auto"
    )
    """
    model_config = ModelConfig(
        type="ollama",
        model_id="ollama/gpt-oss:20b",  # , device="cpu", torch_dtype="auto"
    )

    model = setup_local_model(config=model_config)

    tools_config = MCPConfig()
    tools = setup_mcp_tools(config=tools_config)

    agent = DoclingWritingAgent(model=model, tools=tools)
    agent.run("Write me a document on polymers in food-packaging.")


if __name__ == "__main__":
    main()
