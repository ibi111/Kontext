"""
Runs Docling's DocumentConverter, producing a unified DoclingDocument that
the chunker node consumes directly.
"""

import logging

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption

logger = logging.getLogger(__name__)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(backend=PyPdfiumDocumentBackend),
    }
)


def parse_document(state: dict) -> dict:
    result = converter.convert(state["file_path"])
    doc = result.document

    # Docling's default backend silently drops pages it fails to preprocess
    if result.errors:
        failed_pages = [e.error_message for e in result.errors]
        logger.warning(
            f"Docling reported {len(result.errors)} page-level error(s) "
            f"while converting {state['file_path']}: {failed_pages}. "
            f"Content from these pages may be missing from the output."
        )

    return {
        **state,
        "docling_document": doc,
        "doc_markdown_preview": doc.export_to_markdown()[:1000],
        "parse_errors": [e.error_message for e in result.errors] if result.errors else [],
    }