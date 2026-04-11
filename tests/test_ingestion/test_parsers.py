"""Tests for document parsers."""

import hashlib
import io

import pytest
from docx import Document as DocxDocument
from pypdf import PdfWriter

from graphrag_core.models import ParsedDocument


class TestTextParser:
    @pytest.fixture
    def parser(self):
        from graphrag_core.ingestion.parsers import TextParser
        return TextParser()

    @pytest.mark.asyncio
    async def test_parse_splits_on_double_newline(self, parser):
        source = b"First paragraph.\n\nSecond paragraph."
        result = await parser.parse(source, "text/plain")

        assert isinstance(result, ParsedDocument)
        assert len(result.sections) == 2
        assert result.sections[0].text == "First paragraph."
        assert result.sections[0].heading is None
        assert result.sections[1].text == "Second paragraph."

    @pytest.mark.asyncio
    async def test_parse_sets_metadata(self, parser):
        source = b"Hello world."
        result = await parser.parse(source, "text/plain")

        assert result.metadata.sha256 == hashlib.sha256(source).hexdigest()
        assert result.metadata.doc_type == "text/plain"
        assert result.metadata.title == ""
        assert result.metadata.source == ""
        assert result.metadata.date is None
        assert result.metadata.quarter is None

    @pytest.mark.asyncio
    async def test_parse_rejects_wrong_content_type(self, parser):
        with pytest.raises(ValueError, match="text/plain"):
            await parser.parse(b"hello", "application/pdf")

    @pytest.mark.asyncio
    async def test_parse_single_section(self, parser):
        source = b"Just one section, no double newlines."
        result = await parser.parse(source, "text/plain")
        assert len(result.sections) == 1
        assert result.sections[0].text == "Just one section, no double newlines."

    @pytest.mark.asyncio
    async def test_parse_strips_empty_sections(self, parser):
        source = b"First.\n\n\n\nSecond."
        result = await parser.parse(source, "text/plain")
        assert all(s.text.strip() for s in result.sections)


class TestMarkdownParser:
    @pytest.fixture
    def parser(self):
        from graphrag_core.ingestion.parsers import MarkdownParser
        return MarkdownParser()

    @pytest.mark.asyncio
    async def test_parse_splits_on_headings(self, parser):
        source = b"# Title\n\nContent under title.\n\n## Subtitle\n\nMore content."
        result = await parser.parse(source, "text/markdown")

        assert len(result.sections) == 2
        assert result.sections[0].heading == "Title"
        assert result.sections[0].text == "Content under title."
        assert result.sections[1].heading == "Subtitle"
        assert result.sections[1].text == "More content."

    @pytest.mark.asyncio
    async def test_parse_content_before_first_heading(self, parser):
        source = b"Preamble text.\n\n# First Heading\n\nBody."
        result = await parser.parse(source, "text/markdown")

        assert len(result.sections) == 2
        assert result.sections[0].heading is None
        assert result.sections[0].text == "Preamble text."
        assert result.sections[1].heading == "First Heading"

    @pytest.mark.asyncio
    async def test_parse_sets_metadata(self, parser):
        source = b"# Hello"
        result = await parser.parse(source, "text/markdown")
        assert result.metadata.sha256 == hashlib.sha256(source).hexdigest()
        assert result.metadata.doc_type == "text/markdown"

    @pytest.mark.asyncio
    async def test_parse_rejects_wrong_content_type(self, parser):
        with pytest.raises(ValueError, match="text/markdown"):
            await parser.parse(b"# hello", "application/pdf")

    @pytest.mark.asyncio
    async def test_parse_multiple_heading_levels(self, parser):
        source = b"# H1\n\nText1.\n\n### H3\n\nText3."
        result = await parser.parse(source, "text/markdown")
        assert result.sections[0].heading == "H1"
        assert result.sections[1].heading == "H3"


class TestPdfParser:
    @pytest.fixture
    def parser(self):
        from graphrag_core.ingestion.parsers import PdfParser
        return PdfParser()

    @pytest.mark.asyncio
    async def test_parse_rejects_wrong_content_type(self, parser):
        with pytest.raises(ValueError, match="application/pdf"):
            await parser.parse(b"not a pdf", "text/plain")

    @pytest.mark.asyncio
    async def test_parse_metadata(self, parser):
        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        buf = io.BytesIO()
        writer.write(buf)
        pdf_bytes = buf.getvalue()

        result = await parser.parse(pdf_bytes, "application/pdf")

        assert isinstance(result, ParsedDocument)
        assert result.metadata.doc_type == "application/pdf"
        assert result.metadata.sha256 == hashlib.sha256(pdf_bytes).hexdigest()


class TestDocxParser:
    @pytest.fixture
    def parser(self):
        from graphrag_core.ingestion.parsers import DocxParser
        return DocxParser()

    @staticmethod
    def _make_docx(paragraphs: list[tuple[str | None, str]]) -> bytes:
        """Build a DOCX in memory. Each tuple is (style_name | None, text)."""
        doc = DocxDocument()
        for style, text in paragraphs:
            p = doc.add_paragraph(text)
            if style:
                p.style = doc.styles[style]
        buf = io.BytesIO()
        doc.save(buf)
        return buf.getvalue()

    @pytest.mark.asyncio
    async def test_parse_rejects_wrong_content_type(self, parser):
        with pytest.raises(ValueError, match="application/vnd"):
            await parser.parse(b"not a docx", "text/plain")

    @pytest.mark.asyncio
    async def test_parse_groups_by_headings(self, parser):
        docx_bytes = self._make_docx([
            ("Heading 1", "Introduction"),
            (None, "First body paragraph."),
            (None, "Second body paragraph."),
            ("Heading 1", "Methods"),
            (None, "Methods body."),
        ])
        ct = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        result = await parser.parse(docx_bytes, ct)

        assert len(result.sections) == 2
        assert result.sections[0].heading == "Introduction"
        assert "First body paragraph." in result.sections[0].text
        assert "Second body paragraph." in result.sections[0].text
        assert result.sections[1].heading == "Methods"
        assert "Methods body." in result.sections[1].text

    @pytest.mark.asyncio
    async def test_parse_preamble_before_heading(self, parser):
        docx_bytes = self._make_docx([
            (None, "Preamble text."),
            ("Heading 1", "First Heading"),
            (None, "Body."),
        ])
        ct = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        result = await parser.parse(docx_bytes, ct)

        assert len(result.sections) == 2
        assert result.sections[0].heading is None
        assert result.sections[0].text == "Preamble text."
        assert result.sections[1].heading == "First Heading"

    @pytest.mark.asyncio
    async def test_parse_metadata(self, parser):
        docx_bytes = self._make_docx([(None, "Hello.")])
        ct = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        result = await parser.parse(docx_bytes, ct)

        assert result.metadata.doc_type == ct
        assert result.metadata.sha256 == hashlib.sha256(docx_bytes).hexdigest()
