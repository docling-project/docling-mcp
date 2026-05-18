import json
import os
from pathlib import Path

import pytest

from docling_core.types.doc.document import DoclingDocument

from docling_mcp.logger import setup_logger
from docling_mcp.shared import local_document_cache
from docling_mcp.tools.manipulation import (
    TextSearchOutput,
    search_for_text_in_document_anchors,
)

logger = setup_logger()


def test_search_for_text_in_document_anchors() -> None:
    # Get the golden search results
    source_path = Path("./tests/data/gt_search_results.json")

    golden_results = {}

    if os.path.exists(source_path):
        with open(source_path) as f:
            golden_results = json.load(f)

    GENERATE = True if not golden_results else False

    # Load two documents into the local cache to search across
    file_path = Path("./tests/data/amt_handbook_sample.json")
    doc = DoclingDocument.load_from_json(filename=file_path)
    doc_1_key = "test_doc_1"
    local_document_cache[doc_1_key] = doc

    file_path = Path("./tests/data/lorem_ipsum.docx.json")
    doc = DoclingDocument.load_from_json(filename=file_path)
    doc_2_key = "test_doc_2"
    local_document_cache[doc_2_key] = doc

    # Test exact match searches

    doc_1_result = search_for_text_in_document_anchors(
        document_key=doc_1_key, text="load-carrying nut"
    )

    doc_2_result = search_for_text_in_document_anchors(
        document_key=doc_2_key, text="pellentesque vulputate"
    )

    assert isinstance(doc_1_result, TextSearchOutput)
    assert isinstance(doc_2_result, TextSearchOutput)

    if GENERATE:
        golden_results["exact_match"] = (doc_1_result.result, doc_2_result.result)
    else:
        assert doc_1_result.result == golden_results["exact_match"][0]
        assert doc_2_result.result == golden_results["exact_match"][1]

    # Test keyword match searches

    doc_1_result = search_for_text_in_document_anchors(
        document_key=doc_1_key, text="locking section spring mechanism"
    )

    doc_2_result = search_for_text_in_document_anchors(
        document_key=doc_2_key, text="porttitor varius"
    )

    assert isinstance(doc_1_result, TextSearchOutput)
    assert isinstance(doc_2_result, TextSearchOutput)

    if GENERATE:
        golden_results["keywords_match"] = (doc_1_result.result, doc_2_result.result)
    else:
        assert doc_1_result.result == golden_results["keywords_match"][0]
        assert doc_2_result.result == golden_results["keywords_match"][1]

    # Test no match searches

    doc_1_result = search_for_text_in_document_anchors(
        document_key=doc_1_key, text="Banana Peel"
    )

    doc_2_result = search_for_text_in_document_anchors(
        document_key=doc_2_key, text="Banana Peel"
    )

    assert isinstance(doc_1_result, TextSearchOutput)
    assert isinstance(doc_2_result, TextSearchOutput)

    if GENERATE:
        golden_results["no_match"] = (doc_1_result.result, doc_2_result.result)
    else:
        assert doc_1_result.result == golden_results["no_match"][0]
        assert doc_2_result.result == golden_results["no_match"][1]

    # Test doc key not found

    with pytest.raises(ValueError):
        search_for_text_in_document_anchors(
            document_key="banana_peel", text="load-carrying nut"
        )

    # Save the golden results if generated

    if GENERATE:
        with open(source_path, "w") as f:
            json.dump(golden_results, f, indent=2)
        logger.info(f"Generated golden search results at {source_path}")
