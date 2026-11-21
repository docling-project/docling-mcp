import copy
import logging
import re
from datetime import datetime
from enum import Enum
from io import BytesIO
from typing import ClassVar
import json

from pydantic import BaseModel, Field, validator

from examples.mellea.agent_models import setup_local_session
from examples.mellea.agent.base import DoclingAgentType, BaseDoclingAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DoclingExtractingAgent(BaseDoclingAgent):
    system_prompt_for_editing_document: ClassVar[str] = (
        SYSTEM_PROMPT_FOR_EDITING_DOCUMENT
    )
    system_prompt_for_editing_table: ClassVar[str] = SYSTEM_PROMPT_FOR_EDITING_TABLE

    system_prompt_expert_writer: ClassVar[str] = SYSTEM_PROMPT_EXPERT_WRITER

    def __init__(self, *, model_id: ModelIdentifier, tools: list[Tool]):
        super().__init__(
            agent_type=DoclingAgentType.DOCLING_DOCUMENT_EXTRACTOR,
            model_id=model_id,
            tools=tools,
        )

    def run(self, task: str, document: DoclingDocument, **kwargs) -> DoclingDocument:
        schema: dict = self._extract_schema_from_task(task=task)

        extractions = []
        for item, level in document.iterate_items():
            if isinstance(item, TextItem):
                self._extract_from_text_item(
                    item=item, schema=schema, extractions=extractions
                )

        return document

    def _extract_schema_from_task(self, task: str) -> dict:
        return {}
