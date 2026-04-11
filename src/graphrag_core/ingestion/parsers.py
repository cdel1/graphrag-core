"""BB1: Document parsers for PDF, DOCX, plain text, and Markdown."""

from __future__ import annotations

import hashlib
import re

from graphrag_core.models import DocumentMetadata, ParsedDocument, TextSection


def _metadata(source: bytes, doc_type: str) -> DocumentMetadata:
    return DocumentMetadata(
        title="",
        source="",
        doc_type=doc_type,
        date=None,
        quarter=None,
        sha256=hashlib.sha256(source).hexdigest(),
    )


class TextParser:
    """Parses plain text into sections split on double newlines."""

    async def parse(self, source: bytes, content_type: str) -> ParsedDocument:
        if content_type != "text/plain":
            raise ValueError(f"TextParser expects content_type 'text/plain', got '{content_type}'")

        text = source.decode("utf-8")
        raw_sections = text.split("\n\n")
        sections = [
            TextSection(heading=None, text=s.strip())
            for s in raw_sections
            if s.strip()
        ]
        return ParsedDocument(sections=sections, metadata=_metadata(source, content_type))


class MarkdownParser:
    """Parses Markdown into sections split on headings."""

    _HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)", re.MULTILINE)

    async def parse(self, source: bytes, content_type: str) -> ParsedDocument:
        if content_type != "text/markdown":
            raise ValueError(f"MarkdownParser expects content_type 'text/markdown', got '{content_type}'")

        text = source.decode("utf-8")
        sections: list[TextSection] = []
        last_end = 0
        last_heading: str | None = None

        for match in self._HEADING_RE.finditer(text):
            before = text[last_end:match.start()].strip()
            if before or last_heading is not None:
                sections.append(TextSection(heading=last_heading, text=before))
            elif before:
                sections.append(TextSection(heading=None, text=before))

            last_heading = match.group(2).strip()
            last_end = match.end()

        trailing = text[last_end:].strip()
        if trailing or last_heading is not None:
            sections.append(TextSection(heading=last_heading, text=trailing))

        return ParsedDocument(sections=sections, metadata=_metadata(source, content_type))


class PdfParser:
    """Parses PDF bytes into one TextSection per page."""

    async def parse(self, source: bytes, content_type: str) -> ParsedDocument:
        raise NotImplementedError("PdfParser will be implemented in Task 4")


class DocxParser:
    """Parses DOCX bytes into sections grouped by headings."""

    async def parse(self, source: bytes, content_type: str) -> ParsedDocument:
        raise NotImplementedError("DocxParser will be implemented in Task 4")
