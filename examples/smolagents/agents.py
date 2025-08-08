import re
import logging

from io import BytesIO
from abc import ABC, abstractmethod

from enum import Enum
from pydantic import BaseModel, Field, validator

from typing import ClassVar

from smolagents.models import MessageRole, ChatMessage, Model
from smolagents import (
    MCPClient,
    ToolCollection,
    Tool,
)

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.document import (
    ConversionResult,
)
from docling_core.types.io import DocumentStream
from docling.document_converter import DocumentConverter

from docling_core.types.doc.document import (
    ContentLayer,
    DoclingDocument,
    GroupItem,
    TitleItem,
    SectionHeaderItem,
    TextItem,
    ListItem,
    LevelNumber,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from examples.smolagents.agent_model import ModelConfig, setup_local_model
from examples.smolagents.agent_tools import MCPConfig, setup_mcp_tools


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

    class Config:
        arbitrary_types_allowed = True  # Needed for complex types like Model

    @abstractmethod
    def run(self, task: str, **kwargs) -> str:
        return


class DoclingWritingAgent(BaseDoclingAgent):
    task_analysis: DoclingDocument = DoclingDocument(name=f"report")

    system_prompt_for_task_analysis: ClassVar[
        str
    ] = """You are an expert planner that needs to make a plan to write a document. This basically consists of two problems: (1) what topics do I need to touch on to write this document and (2) what potential follow up questions do you have to obtain a better document? Provide your answer in markdown as a nested list with the following template

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

    system_prompt_for_outline: ClassVar[
        str
    ] = """You are an expert writer that needs to make an outline for a document, i.e. the overall structure of the document in terms of section-headers, text, lists, tables and figures. This outline can be represented as a markdown document. The goal is to have structure of the document with all its items and to provide a 1 sentence summary of each item.   

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
    
## References

list: <1 sentence summary of what the list enumerates>
```

Make sure that the Markdown outline is always enclosed in ```markdown <markdown-content> ```!     
"""

    system_prompt_expert_writer: ClassVar[
        str
    ] = """You are an expert writer that needs to write a single paragraph, table
or nested list based on a summary. Really stick to the summary and be specific, but do not write on adjacent topics    
"""

    def __init__(self, *, model: Model, tools: list[Tool]):
        super().__init__(
            agent_type=DoclingAgentType.DOCLING_DOCUMENT_WRITER,
            model=model,
            tools=tools,
            chat_history=[],
        )

    def _get_system_prompt(self) -> ChatMessage:
        system_prompt = ChatMessage(
            role=MessageRole.SYSTEM,
            content=[
                {
                    "type": "text",
                    "text": "You are an expert writer. You will be asked to write on a variety on topics and I expect you to continuously write in strict MarkDown format.",
                }
            ],
        )
        return system_prompt

    def run(self, task: str, **kwargs):
        # self._analyse_task_for_topics_and_followup_questions(task=task)

        # self._analyse_task_for_final_destination(task=task)

        document: DoclingDocument = self._make_outline_for_writing(task=task)

        document = self._populate_document_with_content(task=task, document=document)

        print(document.export_to_markdown(text_width=72))

    def _analyse_task_for_topics_and_followup_questions(self, *, task: str):
        chat_messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=[
                    {
                        "type": "text",
                        "text": self.system_prompt_for_task_analysis,
                    }
                ],
            ),
            ChatMessage(
                role=MessageRole.USER,
                content=[{"type": "text", "text": f"{task}"}],
            ),
        ]

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
        chat_messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=[
                    {
                        "type": "text",
                        "text": self.system_prompt_for_outline,
                    }
                ],
            ),
            ChatMessage(
                role=MessageRole.USER,
                content=[{"type": "text", "text": f"{task}"}],
            ),
        ]

        output = self.model.generate(messages=chat_messages)

        self.chat_history.extend(chat_messages)
        self.chat_history.append(output)

        results = self._analyse_output_into_docling_document(message=output)
        assert len(results) == 1, (
            "We only want to see a single response from the initial task analysis"
        )

        document = results[0]
        return document

    def _populate_document_with_content(self, *, task: str, document: DoclingDocument):
        for item, level in document.iterate_items(with_groups=True):
            if isinstance(item, TitleItem) or isinstance(item, SectionHeaderItem):
                logger.info(f"starting in {item.text}")
            elif isinstance(item, TextItem):
                if item.text.startswith("paragraph:"):
                    summary = item.text.replace("paragraph:", "").strip()
                    logger.info(f"need to write a paragraph: {summary})")
                    content = self._write_paragraph(
                        summary=summary, item_type="paragraph"
                    )

                    item.text = content

                elif item.text.startswith("table:"):
                    summary = item.text.replace("table:", "").strip()
                    logger.info(f"need to write a table: {summary}")

        return document

    def _analyse_output_into_docling_document(
        self, message: ChatMessage, language: str = "markdown"
    ) -> list[DoclingDocument]:
        def extract_code_blocks(text, language: str):
            pattern = rf"```{language}\s*(.*?)\s*```"
            matches = re.findall(pattern, text, re.DOTALL)
            return matches

        print(
            f"content: \n\n--------------------\n{message.content}\n--------------------\n"
        )

        converter = DocumentConverter(allowed_formats=[InputFormat.MD])

        result = []
        for mtch in extract_code_blocks(message.content, language=language):
            md_doc: str = mtch
            print("md-doc:\n\n", md_doc)

            buff = BytesIO(md_doc.encode("utf-8"))
            doc_stream = DocumentStream(name="tmp.md", stream=buff)

            conv_result: ConversionResult = converter.convert(doc_stream)
            result.append(conv_result.document)

        logger.warning(f"#-results: {len(result)}")

        return result

    def _write_paragraph(
        self, summary: str, item_type: str, task: str = "", hierarchy: list[str] = []
    ) -> str:
        chat_messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=[
                    {
                        "type": "text",
                        "text": self.system_prompt_expert_writer,
                    }
                ],
            ),
            ChatMessage(
                role=MessageRole.USER,
                content=[
                    {
                        "type": "text",
                        "text": f"write me a single {item_type} that expands the following summary: {summary}",
                    }
                ],
            ),
        ]

        output = self.model.generate(messages=chat_messages)
        return output.content


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
