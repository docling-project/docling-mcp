from docling_mcp.logger import setup_logger
from docling_mcp.shared import mcp
from docling_mcp.tools.conversion import (  # noqa: F401
    convert_pdf_document_into_json_docling_document_from_uri_path,
    is_document_in_local_cache,
)
from docling_mcp.tools.generation import (  # noqa: F401
    add_listitem_to_list_in_docling_document,
    add_paragraph_to_docling_document,
    add_section_heading_to_docling_document,
    add_title_to_docling_document,
    close_list_in_docling_document,
    create_new_docling_document,
    export_docling_document_to_markdown,
    open_list_in_docling_document,
    save_docling_document,
)
from docling_mcp.tools.manipulation import (  # noqa: F401
    delete_document_items_at_anchors,
    get_overview_of_document_anchors,
    get_text_of_document_item_at_anchor,
    update_text_of_document_item_at_anchor,
)

if __name__ == "__main__":
    # Create a default project logger
    logger = setup_logger()
    logger.info("starting up Docling MCP-server ...")

    # Initialize and run the server
    mcp.run(transport="stdio")
