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
    GroupItem,
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
    SYSTEM_PROMPT_FOR_DOCUMENT_ITEMS,
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
    DOCLING_DOCUMENT_EDITOR = "editor"

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
    model_id: ModelIdentifier
    tools: list[Tool]

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

    @staticmethod
    def find_markdown_code_block(text: str) -> str | None:
        """
        Check if a string matches the pattern ```markdown(.*)?```
        """
        pattern = r"```markdown(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1) if match else None

    @staticmethod
    def has_markdown_code_block(text: str) -> bool:
        """
        Check if a string contains a markdown code block pattern anywhere in the text
        """
        logger.info(f"testing has_markdown_code_block for {text[0:64]}")
        return DoclingWritingAgent.find_markdown_code_block(text) is not None

    @staticmethod
    def find_html_code_block(text: str) -> str | None:
        """
        Check if a string matches the pattern ```html(.*)?```
        """
        pattern = r"```html(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1) if match else None

    @staticmethod
    def has_html_code_block(text: str) -> bool:
        """
        Check if a string contains a html code block pattern anywhere in the text
        """
        logger.info(f"testing has_html_code_block for {text[0:64]}")
        return DoclingWritingAgent.find_html_code_block(text) is not None

    @staticmethod
    def find_outline(text: str) -> DoclingDocument | None:
        """
        Check if a string matches the pattern ```markdown(.*)?```
        """
        starts = ["paragraph", "list", "table", "figure", "picture"]

        md = DoclingWritingAgent.find_markdown_code_block(text)

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
                    # print(f"unsupported item: {item}")
                    continue

            if len(lines) > 0:
                message = f"Every content line should start with one out of the following choices: {starts}. The following lines need to be updated: {'\n'.join(lines)}"
                print(message)
                return None
            else:
                return conv.document
        else:
            return None

    @staticmethod
    def validate_outline_format(text: str) -> bool:
        logger.info(f"testing validate_outline_format for {text[0:64]}")
        return DoclingWritingAgent.find_outline(text) is not None

    @staticmethod
    def convert_markdown_to_docling_document(text: str) -> DoclingDocument | None:
        try:
            converter = DocumentConverter(allowed_formats=[InputFormat.MD])

            buff = BytesIO(text.encode("utf-8"))
            doc_stream = DocumentStream(name="tmp.md", stream=buff)

            conv: ConversionResult = converter.convert(doc_stream)

            if conv.status == ConversionStatus.SUCCESS:
                return conv.document
        except Exception as exc:
            # logger.error(exc)
            return None

        return None

    @staticmethod
    def convert_html_to_docling_document(text: str) -> DoclingDocument | None:
        text_ = DoclingWritingAgent.find_html_code_block(text)
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
            print(exc)
            return None

        return None

    @staticmethod
    def validate_markdown_to_docling_document(text: str) -> bool:
        logger.info(f"testing validate_markdown_docling_document for {text[0:64]}")
        return (
            DoclingWritingAgent.convert_markdown_to_docling_document(text) is not None
        )

    @staticmethod
    def validate_html_to_docling_document(text: str) -> bool:
        logger.info(f"testing validate_html_docling_document for {text[0:64]}")
        return DoclingWritingAgent.convert_html_to_docling_document(text) is not None

    def __init__(self, *, model_id: ModelIdentifier, tools: list[Tool]):
        super().__init__(
            agent_type=DoclingAgentType.DOCLING_DOCUMENT_WRITER,
            model_id=model_id,
            tools=tools,
        )

    def run(self, task: str, **kwargs) -> DoclingDocument:
        # self._analyse_task_for_topics_and_followup_questions(task=task)

        # self._analyse_task_for_final_destination(task=task)

        # Plan an outline for the document
        outline: DoclingDocument = self._make_outline_for_writing(task=task)

        # Write the actual document item by item
        document: DoclingDocument = self._populate_document_with_content(
            task=task, outline=outline
        )

        return document

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

    def _make_outline_for_writing(
        self, *, task: str, loop_budget: int = 5
    ) -> DoclingDocument:
        m = setup_local_session(
            model_id=self.model_id, system_prompt=self.system_prompt_for_outline
        )

        answer = m.instruct(
            f"{task}",
            requirements=[
                # "The resulting output should satisfy the following regex ```markdown(.*)?```"
                Requirement(
                    description="Put the resulting markdown outline in the format ```markdown <insert-content>```",
                    validation_fn=simple_validate(
                        DoclingWritingAgent.has_markdown_code_block
                    ),
                ),
                # "The resulting outline should have a specific format: start with `paragraph: `,
                # `table: `, `picture: ` or `list: `.
                Requirement(
                    description="The resulting outline should be in markdown format. If not a title or subheading, start each line with `paragraph: `, `table: `, `picture: ` or `list: ` followed by a single sentence summary.",
                    validation_fn=simple_validate(
                        DoclingWritingAgent.validate_outline_format
                    ),
                ),
            ],
            # user_variables={"name": name, "notes": notes},
            strategy=RejectionSamplingStrategy(loop_budget=loop_budget),
        )

        outline = DoclingWritingAgent.find_outline(text=answer.value)
        print(outline.export_to_markdown())

        return outline

    def _update_document_with_content(
        self, *, document: DoclingDocument, content: DoclingDocument
    ) -> DoclingDocument:
        to_item: dict[str, NodeItem] = {}

        for item, level in content.iterate_items(with_groups=True):
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

            elif isinstance(item, TableItem):
                if item.parent and item.parent.cref in to_item:
                    if len(item.captions) > 0:
                        caption = document.add_text(
                            label=DocItemLabel.CAPTION, text=item.captions[0].text
                        )
                        te = document.add_table(
                            data=item.data,
                            caption=caption,
                        )
                        to_item[item.self_ref] = te
                    else:
                        te = document.add_table(
                            data=item.data,
                            # caption=caption,
                        )
                        to_item[item.self_ref] = te
                else:
                    print("skipping: ", item)

            else:
                print("skipping: ", item)

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
                    new_content = self._write_paragraph(
                        summary=summary, hierarchy=headers
                    )
                    document = self._update_document_with_content(
                        document=document, content=new_content
                    )

                elif item.text.startswith("list:"):
                    summary = item.text.replace("list:", "").strip()
                    logger.info(f"need to write a list: {summary}")

                    new_content = self._write_list(summary=summary, hierarchy=headers)
                    document = self._update_document_with_content(
                        document=document, content=new_content
                    )

                elif item.text.startswith("table:"):
                    summary = item.text.replace("table:", "").strip()
                    logger.info(f"need to write a table: {summary}")

                    new_content = self._write_table(summary=summary, hierarchy=headers)

                    document = self._update_document_with_content(
                        document=document, content=new_content
                    )

                elif item.text.startswith("picture:") or item.text.startswith(
                    "figure:"
                ):
                    summary = (
                        item.text.replace("picture:", "").replace("figure:", "").strip()
                    )

                    caption = document.add_text(
                        label=DocItemLabel.CAPTION, text=summary
                    )
                    document.add_picture(caption=caption)

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
        self,
        summary: str,
        task: str = "",
        hierarchy: dict[int, str] = {},
        loop_budget: int = 5,
    ) -> str:
        context = ""
        for level, header in hierarchy.items():
            context += "#" * (level + 1) + header + "\n"

        if len(context) > 0:
            context = rf"Given the current context in the document:\n\n```markdown\n{context}```\n\n"

        m = setup_local_session(
            model_id=self.model_id,
            system_prompt=self.system_prompt_expert_writer,
        )

        prompt = f"{context}Write me a single paragraph that expands the following summary: {summary}"
        logger.info(f"prompt: {prompt}")

        answer = m.instruct(
            prompt,
            requirements=[
                Requirement(
                    description="The resulting markdown paragraph should use latex notation for superscript, subscript or inline equations. This means that every superscript, subscript and inline equation in must start and end with a $ sign.",
                    validation_fn=simple_validate(
                        DoclingWritingAgent.validate_markdown_to_docling_document
                    ),
                ),
            ],
            strategy=RejectionSamplingStrategy(loop_budget=loop_budget),
        )

        result = DoclingWritingAgent.convert_markdown_to_docling_document(
            text=answer.value
        )

        return result

    def _write_list(
        self,
        summary: str,
        task: str = "",
        hierarchy: dict[int, str] = {},
        loop_budget: int = 5,
    ) -> DoclingDocument | None:
        context = ""
        for level, header in hierarchy.items():
            context += "#" * (level + 1) + header + "\n"

        m = setup_local_session(
            model_id=self.model_id,
            system_prompt=self.system_prompt_expert_writer,
        )

        prompt = f"Given the current context in the document:\n\n```{context}```\n\nwrite me a list (can be nested) in markdown that expands the following summary: {summary}"
        logger.info(f"prompt: {prompt}")

        answer = m.instruct(
            prompt,
            requirements=[
                Requirement(
                    description="The resulting markdown list should use latex notation for superscript, subscript or inline equations. This means that every superscript, subscript and inline equation in must start and end with a $ sign.",
                    validation_fn=simple_validate(
                        DoclingWritingAgent.validate_markdown_to_docling_document
                    ),
                ),
            ],
            strategy=RejectionSamplingStrategy(loop_budget=loop_budget),
        )

        result = DoclingWritingAgent.convert_markdown_to_docling_document(
            text=answer.value
        )

        return result

    def _write_table(
        self,
        summary: str,
        task: str = "",
        hierarchy: dict[int, str] = {},
        loop_budget: int = 5,
    ) -> str:
        context = ""
        for level, header in hierarchy.items():
            context += "#" * (level + 1) + header + "\n"

        m = setup_local_session(
            model_id=self.model_id,
            system_prompt=self.system_prompt_expert_writer,
        )

        prompt = f"Given the current context in the document:\n\n```{context}```\n\nwrite me a single HTML table that expands the following summary: {summary}"
        logger.info(f"prompt: {prompt}")

        answer = m.instruct(
            prompt,
            requirements=[
                Requirement(
                    description="Put the resulting HTML table in the format ```html <insert-content>```",
                    validation_fn=simple_validate(
                        DoclingWritingAgent.has_html_code_block
                    ),
                ),
                Requirement(
                    description="The HTML table should have a valid formatting.",
                    validation_fn=simple_validate(
                        DoclingWritingAgent.validate_html_to_docling_document
                    ),
                ),
            ],
            strategy=RejectionSamplingStrategy(loop_budget=loop_budget),
        )
        # print(answer.value)

        result = DoclingWritingAgent.convert_html_to_docling_document(text=answer.value)

        return result


class DoclingEditingAgent(BaseDoclingAgent):
    system_prompt_for_document_items: ClassVar[str] = SYSTEM_PROMPT_FOR_DOCUMENT_ITEMS

    @staticmethod
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

    @staticmethod
    def has_crefs(text: str) -> bool:
        return len(find_crefs) > 0

    def __init__(self, *, model_id: ModelIdentifier, tools: list[Tool]):
        super().__init__(
            agent_type=DoclingAgentType.DOCLING_DOCUMENT_EDITOR,
            model_id=model_id,
            tools=tools,
        )

    def run(self, task: str, document: DoclingDocument, **kwargs) -> DoclingDocument:
        refs = self._identify_document_items(task=task, document=document)

        # self.transform_document_items(task=task, document=document, refs=refs)

        self._update_content_of_document_items(task=task, document=document, refs=refs)

    def _identify_document_items(
        self,
        task: str,
        document: DoclingDocument,
        loop_budget: int = 5,
    ) -> list[RefItem]:
        label_counter: dict[DocItemLabel, int] = {
            DocItemLabel.TABLE: 0,
            DocItemLabel.PICTURE: 0,
            DocItemLabel.TEXT: 0,
        }

        lines = []
        for item, level in document.iterate_items(with_groups=True):
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

        context = rf"""Given the current outline of the document:
```
{outline}
```"""

        identification = rf"""To accomplish the following task:

{task}

We first need to:
    - identify from the outline all the document items that are relevant
    - plan the operations needed to update the document

Now, provide me the operations and their references to execute the task!"""

        # prompt = f"{context}Return all references (in a comma seperated list) that are needed to accomplish this task: {task}. Do NOT return references that are not relevant!"

        prompt = f"{context}{identification}"
        logger.info(f"prompt:\n\n{prompt}")

        m = setup_local_session(
            model_id=self.model_id,
            system_prompt=self.system_prompt_for_document_items,
        )

        """
        answer = m.instruct(
            prompt,
            requirements=[
                Requirement(
                    description="The result should be a plain list of references with the format `#/<label>/<int>`.",
                    validation_fn=simple_validate(
                        DoclingWritingAgent.has_crefs
                    ),
                ),
            ],
            strategy=RejectionSamplingStrategy(loop_budget=loop_budget),
        )
        """
        answer = m.instruct(
            prompt,
            strategy=RejectionSamplingStrategy(loop_budget=loop_budget),
        )
        print(f"""
        ======================
        answer: `{answer.value}`
        ======================
        """)

        result = DoclingWritingAgent.find_crefs(text=answer.value)

        return result

    def _update_content_of_document_items(
        self, task: str, document: DoclingDocument, refs: list[RefItem]
    ):
        for ref in refs:
            item = ref.resolve(document)

            if isinstance(item, TableItem):
                self._update_content_of_table(task=task, document=document, table=item)

            elif isinstance(item, TitleItem):
                self._update_content_of_title(task=task, document=document, table=item)

    def _update_content_of_table(
        self, task: str, document: DoclingDocument, table: TableItem
    ):
        print("_update_content_of_table")
