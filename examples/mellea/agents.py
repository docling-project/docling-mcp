import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from examples.mellea.agent.base import DoclingAgentType, BaseDoclingAgent
from examples.mellea.agent.writer import DoclingWritingAgent
from examples.mellea.agent.editor import DoclingEditingAgent
