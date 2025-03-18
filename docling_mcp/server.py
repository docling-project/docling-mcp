# from typing import Any
# import httpx

#from docling_mcp.tools.conversion import mcp as tools_mcp  # Import the mcp instance, not just the function

#from mcp.server.fastmcp import FastMCP

from docling_mcp.shared import mcp
from docling_mcp.tools import conversion  # This imports your tools module, registering the tools


"""
from typing import Tuple, Dict, Annotated

from mcp.shared.exceptions import McpError
from mcp.types import ErrorData, INTERNAL_ERROR, INVALID_PARAMS

from docling_core.types.doc import DoclingDocument

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfBackend,
    PdfPipelineOptions, 
    OcrEngine, 
    EasyOcrOptions
)
from docling.datamodel.settings import settings
from docling.utils.accelerator_utils import AcceleratorDevice
from docling.utils import accelerator_utils

from docling_mcp.docling_cache import get_cache_dir, get_cache_key



from docling_mcp.docling_settings import configure_accelerator

from docling_mcp.tools.conversion import is_document_in_local_cache, \
    convert_pdf_document_into_json_docling_document_from_uri_path, \
    convert_attachments_into_docling_document
"""

from docling_mcp.logger import setup_logger

# Create a default project logger
logger = setup_logger()

# Initialize FastMCP server
# configure_accelerator()
# mcp = FastMCP("docling")

# local_document_cache: Dict[str, DoclingDocument] = {}


# Register tools from the other module
# mcp.register_tools_from(tools_mcp)
    
if __name__ == "__main__":
    logger.info("starting up Docling MCP-server ...")
    
    # Initialize and run the server
    mcp.run(transport='stdio')     
