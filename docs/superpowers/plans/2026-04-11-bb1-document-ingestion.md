# BB1: Document Ingestion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement default document ingestion — parsers (PDF, DOCX, text, Markdown), token-based chunker, and IngestionPipeline orchestrator.

**Architecture:** Four `DocumentParser` implementations, one `TokenChunker`, and one `IngestionPipeline` class that wires them together. All live under `src/graphrag_core/ingestion/` as a package. An `IngestionPipeline` Protocol is added to `interfaces.py` alongside the existing BB1 interfaces.

**Tech Stack:** Python 3.12+, Pydantic v2, pypdf, python-docx, pytest

---

## File Map

| Action | File | Responsibility |
|---|---|---|
| Create | `src/graphrag_core/ingestion/__init__.py` | Re-exports all public names |
| Create | `src/graphrag_core/ingestion/parsers.py` | PdfParser, DocxParser, TextParser, MarkdownParser |
| Create | `src/graphrag_core/ingestion/chunker.py` | TokenChunker |
| Create | `src/graphrag_core/ingestion/pipeline.py` | IngestionPipeline concrete class |
| Create | `tests/test_ingestion/__init__.py` | Test package marker |
| Create | `tests/test_ingestion/test_parsers.py` | Parser unit tests |
| Create | `tests/test_ingestion/test_chunker.py` | Chunker unit tests |
| Create | `tests/test_ingestion/test_pipeline.py` | Pipeline wiring tests |
| Modify | `src/graphrag_core/interfaces.py:20` | Add IngestionPipeline Protocol |
| Modify | `src/graphrag_core/__init__.py:5-37` | Add IngestionPipeline + ingestion re-exports |
| Modify | `pyproject.toml:7` | Add pypdf, python-docx dependencies |

---

### Task 1: Add dependencies and install

**Files:**
- Modify: `pyproject.toml:7`

- [ ] **Step 1: Add pypdf and python-docx to pyproject.toml**

In `pyproject.toml`, change the `dependencies` line:

```python
# before
dependencies = ["pydantic>=2.0"]

# after
dependencies = ["pydantic>=2.0", "pypdf>=4.0", "python-docx>=1.0"]
```

- [ ] **Step 2: Add pytest-asyncio to dev dependencies**

In `pyproject.toml`, change the dev dependency group:

```python
# before
dev = [
    "pytest>=9.0.3",
]

# after
dev = [
    "pytest>=9.0.3",
    "pytest-asyncio>=0.24",
]
```

- [ ] **Step 3: Install dependencies**

Run: `cd /Users/dinoceli/Developer/tessera/repos/graphrag-core && uv sync`

Expected: Clean install with pypdf, python-docx, and pytest-asyncio added.

- [ ] **Step 4: Verify imports work**

Run: `cd /Users/dinoceli/Developer/tessera/repos/graphrag-core && uv run python -c "import pypdf; import docx; import pytest_asyncio; print('OK')"`

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add pypdf, python-docx, and pytest-asyncio dependencies"
```

---

### Task 2: Add IngestionPipeline Protocol to interfaces.py

**Files:**
- Modify: `src/graphrag_core/interfaces.py:20`
- Modify: `src/graphrag_core/__init__.py:5-37`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ingestion/__init__.py` (empty file) and `tests/test_ingestion/test_pipeline_protocol.py`:

```python
"""Test that IngestionPipeline Protocol exists and is importable."""

from graphrag_core.interfaces import IngestionPipeline


def test_ingestion_pipeline_protocol_is_runtime_checkable():
    import typing
    assert hasattr(IngestionPipeline, "__protocol_attrs__") or issubclass(
        IngestionPipeline, typing.Protocol
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/dinoceli/Developer/tessera/repos/graphrag-core && uv run pytest tests/test_ingestion/test_pipeline_protocol.py -v`

Expected: FAIL with `ImportError: cannot import name 'IngestionPipeline' from 'graphrag_core.interfaces'`

- [ ] **Step 3: Add IngestionPipeline Protocol to interfaces.py**

In `src/graphrag_core/interfaces.py`, add after the existing BB1 section (after `EmbeddingModel`, before BB2):

```python
@runtime_checkable
class IngestionPipeline(Protocol):
    """Orchestrates parsing, chunking, and optional embedding."""

    async def ingest(
        self,
        source: bytes,
        content_type: str,
        config: ChunkConfig | None = None,
    ) -> list[DocumentChunk]: ...
```

Also add `ChunkConfig` to the imports at the top of `interfaces.py`:

```python
from graphrag_core.models import (
    AuditTrail,
    ChunkConfig,  # <-- add this
    DocumentChunk,
    ...
)
```

- [ ] **Step 4: Add IngestionPipeline to __init__.py re-exports**

In `src/graphrag_core/__init__.py`, add `IngestionPipeline` to the imports from `interfaces` and to `__all__`:

```python
from graphrag_core.interfaces import (
    Chunker,
    DocumentParser,
    EmbeddingModel,
    ExtractionEngine,
    GraphStore,
    IngestionPipeline,  # <-- add this
    SearchEngine,
)
```

```python
__all__ = [
    "Chunker",
    "DocumentParser",
    "EmbeddingModel",
    "ExtractionEngine",
    "GraphStore",
    "IngestionPipeline",  # <-- add this
    "SearchEngine",
    ...
]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /Users/dinoceli/Developer/tessera/repos/graphrag-core && uv run pytest tests/test_ingestion/test_pipeline_protocol.py -v`

Expected: PASS

- [ ] **Step 6: Run full test suite**

Run: `cd /Users/dinoceli/Developer/tessera/repos/graphrag-core && uv run pytest tests/ -v`

Expected: All tests pass (existing 31 + 1 new).

- [ ] **Step 7: Commit**

```bash
git add src/graphrag_core/interfaces.py src/graphrag_core/__init__.py tests/test_ingestion/
git commit -m "feat: add IngestionPipeline Protocol to interfaces"
```

---

### Task 3: TextParser and MarkdownParser

Starting with the simplest parsers — no external dependencies needed.

**Files:**
- Create: `src/graphrag_core/ingestion/__init__.py`
- Create: `src/graphrag_core/ingestion/parsers.py`
- Create: `tests/test_ingestion/test_parsers.py`

- [ ] **Step 1: Write failing tests for TextParser**

Create `tests/test_ingestion/test_parsers.py`:

```python
"""Tests for document parsers."""

import hashlib

import pytest

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/dinoceli/Developer/tessera/repos/graphrag-core && uv run pytest tests/test_ingestion/test_parsers.py -v`

Expected: FAIL — `ModuleNotFoundError: No module named 'graphrag_core.ingestion'`

- [ ] **Step 3: Create the ingestion package and implement TextParser + MarkdownParser**

Create `src/graphrag_core/ingestion/__init__.py`:

```python
"""BB1: Document Ingestion — default implementations."""

from graphrag_core.ingestion.parsers import (
    DocxParser,
    MarkdownParser,
    PdfParser,
    TextParser,
)
from graphrag_core.ingestion.chunker import TokenChunker
from graphrag_core.ingestion.pipeline import IngestionPipeline

__all__ = [
    "DocxParser",
    "IngestionPipeline",
    "MarkdownParser",
    "PdfParser",
    "TextParser",
    "TokenChunker",
]
```

Note: This `__init__.py` imports from all submodules. We'll create stub files for `chunker.py` and `pipeline.py` in this step too, so the import doesn't fail. The parsers that aren't implemented yet (`PdfParser`, `DocxParser`) will be implemented in Task 4.

Create `src/graphrag_core/ingestion/chunker.py` (stub):

```python
"""BB1: Token-based chunker."""
```

Create `src/graphrag_core/ingestion/pipeline.py` (stub):

```python
"""BB1: Ingestion pipeline orchestrator."""
```

Create `src/graphrag_core/ingestion/parsers.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/dinoceli/Developer/tessera/repos/graphrag-core && uv run pytest tests/test_ingestion/test_parsers.py -v`

Expected: All 10 tests PASS.

- [ ] **Step 5: Run full test suite**

Run: `cd /Users/dinoceli/Developer/tessera/repos/graphrag-core && uv run pytest tests/ -v`

Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/graphrag_core/ingestion/ tests/test_ingestion/test_parsers.py
git commit -m "feat: add TextParser and MarkdownParser implementations"
```

---

### Task 4: PdfParser and DocxParser

**Files:**
- Modify: `src/graphrag_core/ingestion/parsers.py`
- Modify: `tests/test_ingestion/test_parsers.py`

- [ ] **Step 1: Write failing tests for PdfParser**

Append to `tests/test_ingestion/test_parsers.py`:

```python
import io
from pypdf import PdfWriter


class TestPdfParser:
    @pytest.fixture
    def parser(self):
        from graphrag_core.ingestion.parsers import PdfParser
        return PdfParser()

    @staticmethod
    def _make_pdf(pages: list[str]) -> bytes:
        writer = PdfWriter()
        for text in pages:
            writer.add_blank_page(width=612, height=792)
            page = writer.pages[-1]
            # Inject text via content stream — minimal valid PDF text
            from pypdf.generic import (
                ArrayObject,
                DecodedStreamObject,
                NameObject,
                NumberObject,
                DictionaryObject,
            )
            # Use annotation-based approach: add text to page
            # Simpler approach: create a real PDF using reportlab-free method
            pass
        # Since injecting raw PDF text streams is complex, use a simpler fixture:
        # build a real minimal PDF with pypdf
        buf = io.BytesIO()
        writer.write(buf)
        return buf.getvalue()

    @pytest.mark.asyncio
    async def test_parse_rejects_wrong_content_type(self, parser):
        with pytest.raises(ValueError, match="application/pdf"):
            await parser.parse(b"not a pdf", "text/plain")

    @pytest.mark.asyncio
    async def test_parse_returns_parsed_document(self, parser):
        # Create a minimal PDF with pypdf
        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        buf = io.BytesIO()
        writer.write(buf)
        pdf_bytes = buf.getvalue()

        result = await parser.parse(pdf_bytes, "application/pdf")

        assert isinstance(result, ParsedDocument)
        assert result.metadata.doc_type == "application/pdf"
        assert result.metadata.sha256 == hashlib.sha256(pdf_bytes).hexdigest()
        # Blank page → one section with empty or whitespace text
        assert len(result.sections) >= 0  # blank pages may be filtered

    @pytest.mark.asyncio
    async def test_parse_sets_page_numbers(self, parser):
        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        writer.add_blank_page(width=612, height=792)
        buf = io.BytesIO()
        writer.write(buf)
        pdf_bytes = buf.getvalue()

        result = await parser.parse(pdf_bytes, "application/pdf")

        for section in result.sections:
            assert section.page is not None
```

**Important:** Testing PDF with real text content is tricky without `reportlab`. The tests above verify structure and metadata. For a better test, use a pre-built PDF fixture. However, keeping it simple: test content-type rejection, metadata, and page numbering with blank pages. The page-text extraction is tested implicitly via integration.

**Revised approach — simpler tests that are more reliable:**

Replace the entire `TestPdfParser` class above with:

```python
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
```

Also add the imports at the top of the file:

```python
import io
from pypdf import PdfWriter
```

- [ ] **Step 2: Write failing tests for DocxParser**

Append to `tests/test_ingestion/test_parsers.py`:

```python
from docx import Document as DocxDocument


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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /Users/dinoceli/Developer/tessera/repos/graphrag-core && uv run pytest tests/test_ingestion/test_parsers.py::TestPdfParser tests/test_ingestion/test_parsers.py::TestDocxParser -v`

Expected: FAIL — `NotImplementedError` from stub implementations.

- [ ] **Step 4: Implement PdfParser**

Replace the `PdfParser` class in `src/graphrag_core/ingestion/parsers.py`:

```python
class PdfParser:
    """Parses PDF bytes into one TextSection per page."""

    async def parse(self, source: bytes, content_type: str) -> ParsedDocument:
        if content_type != "application/pdf":
            raise ValueError(f"PdfParser expects content_type 'application/pdf', got '{content_type}'")

        import io
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(source))
        sections = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                sections.append(TextSection(heading=None, text=text.strip(), page=i + 1))
        return ParsedDocument(sections=sections, metadata=_metadata(source, content_type))
```

- [ ] **Step 5: Implement DocxParser**

Replace the `DocxParser` class in `src/graphrag_core/ingestion/parsers.py`:

```python
class DocxParser:
    """Parses DOCX bytes into sections grouped by headings."""

    _CONTENT_TYPE = (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )

    async def parse(self, source: bytes, content_type: str) -> ParsedDocument:
        if content_type != self._CONTENT_TYPE:
            raise ValueError(
                f"DocxParser expects content_type '{self._CONTENT_TYPE}', got '{content_type}'"
            )

        import io
        from docx import Document

        doc = Document(io.BytesIO(source))
        sections: list[TextSection] = []
        current_heading: str | None = None
        current_lines: list[str] = []

        for para in doc.paragraphs:
            if para.style and para.style.name and para.style.name.startswith("Heading"):
                # Flush previous section
                if current_lines or current_heading is not None:
                    sections.append(TextSection(
                        heading=current_heading,
                        text="\n".join(current_lines).strip(),
                    ))
                current_heading = para.text.strip()
                current_lines = []
            else:
                text = para.text.strip()
                if text:
                    current_lines.append(text)

        # Flush last section
        if current_lines or current_heading is not None:
            sections.append(TextSection(
                heading=current_heading,
                text="\n".join(current_lines).strip(),
            ))

        return ParsedDocument(sections=sections, metadata=_metadata(source, content_type))
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /Users/dinoceli/Developer/tessera/repos/graphrag-core && uv run pytest tests/test_ingestion/test_parsers.py -v`

Expected: All parser tests PASS.

- [ ] **Step 7: Run full test suite**

Run: `cd /Users/dinoceli/Developer/tessera/repos/graphrag-core && uv run pytest tests/ -v`

Expected: All tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/graphrag_core/ingestion/parsers.py tests/test_ingestion/test_parsers.py
git commit -m "feat: add PdfParser and DocxParser implementations"
```

---

### Task 5: TokenChunker

**Files:**
- Modify: `src/graphrag_core/ingestion/chunker.py`
- Create: `tests/test_ingestion/test_chunker.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_ingestion/test_chunker.py`:

```python
"""Tests for TokenChunker."""

import pytest

from graphrag_core.models import (
    ChunkConfig,
    DocumentChunk,
    DocumentMetadata,
    ParsedDocument,
    TextSection,
)


def _doc(texts: list[str], sha: str = "abc123def456") -> ParsedDocument:
    """Build a minimal ParsedDocument for testing."""
    return ParsedDocument(
        sections=[TextSection(heading=None, text=t) for t in texts],
        metadata=DocumentMetadata(
            title="", source="", doc_type="text/plain",
            date=None, quarter=None, sha256=sha,
        ),
    )


class TestTokenChunker:
    @pytest.fixture
    def chunker(self):
        from graphrag_core.ingestion.chunker import TokenChunker
        return TokenChunker()

    def test_single_chunk_below_max(self, chunker):
        doc = _doc(["one two three"])
        config = ChunkConfig(max_tokens=10, overlap=0)
        chunks = chunker.chunk(doc, config)

        assert len(chunks) == 1
        assert chunks[0].text == "one two three"
        assert chunks[0].position == 0
        assert chunks[0].id == "abc123def456-0"
        assert chunks[0].embedding is None
        assert chunks[0].chunk_type == "text"

    def test_splits_at_max_tokens(self, chunker):
        doc = _doc(["a b c d e f g h i j"])
        config = ChunkConfig(max_tokens=5, overlap=0)
        chunks = chunker.chunk(doc, config)

        assert len(chunks) == 2
        assert chunks[0].text == "a b c d e"
        assert chunks[1].text == "f g h i j"
        assert chunks[0].position == 0
        assert chunks[1].position == 1

    def test_overlap(self, chunker):
        doc = _doc(["a b c d e f g h i j"])
        config = ChunkConfig(max_tokens=5, overlap=2)
        chunks = chunker.chunk(doc, config)

        assert len(chunks) == 3
        assert chunks[0].text == "a b c d e"
        assert chunks[1].text == "d e f g h"
        assert chunks[2].text == "g h i j"

    def test_multiple_sections_concatenated(self, chunker):
        doc = _doc(["a b c", "d e f"])
        config = ChunkConfig(max_tokens=4, overlap=0)
        chunks = chunker.chunk(doc, config)

        assert len(chunks) == 2
        assert chunks[0].text == "a b c d"
        assert chunks[1].text == "e f"

    def test_empty_document(self, chunker):
        doc = _doc([])
        config = ChunkConfig(max_tokens=10, overlap=0)
        chunks = chunker.chunk(doc, config)
        assert chunks == []

    def test_page_number_from_section(self, chunker):
        doc = ParsedDocument(
            sections=[TextSection(heading=None, text="word " * 3, page=5)],
            metadata=DocumentMetadata(
                title="", source="", doc_type="text/plain",
                date=None, quarter=None, sha256="abc123def456",
            ),
        )
        config = ChunkConfig(max_tokens=10, overlap=0)
        chunks = chunker.chunk(doc, config)
        assert chunks[0].page == 5

    def test_id_uses_sha_prefix(self, chunker):
        doc = _doc(["hello world"], sha="abcdef1234567890abcdef")
        config = ChunkConfig(max_tokens=10, overlap=0)
        chunks = chunker.chunk(doc, config)
        assert chunks[0].id == "abcdef123456-0"

    def test_default_config(self, chunker):
        config = ChunkConfig()
        assert config.max_tokens == 512
        assert config.overlap == 50
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/dinoceli/Developer/tessera/repos/graphrag-core && uv run pytest tests/test_ingestion/test_chunker.py -v`

Expected: FAIL — `ImportError: cannot import name 'TokenChunker' from 'graphrag_core.ingestion.chunker'`

- [ ] **Step 3: Implement TokenChunker**

Replace contents of `src/graphrag_core/ingestion/chunker.py`:

```python
"""BB1: Token-based chunker."""

from __future__ import annotations

from graphrag_core.models import ChunkConfig, DocumentChunk, ParsedDocument


class TokenChunker:
    """Splits documents into chunks by whitespace token count with overlap."""

    def chunk(self, doc: ParsedDocument, config: ChunkConfig) -> list[DocumentChunk]:
        words: list[tuple[str, int | None]] = []
        for section in doc.sections:
            for word in section.text.split():
                words.append((word, section.page))

        if not words:
            return []

        sha_prefix = doc.metadata.sha256[:12]
        chunks: list[DocumentChunk] = []
        start = 0
        position = 0

        while start < len(words):
            end = min(start + config.max_tokens, len(words))
            chunk_words = words[start:end]
            text = " ".join(w for w, _ in chunk_words)
            page = chunk_words[0][1] if chunk_words else None

            chunks.append(DocumentChunk(
                id=f"{sha_prefix}-{position}",
                text=text,
                page=page,
                position=position,
            ))

            position += 1
            step = config.max_tokens - config.overlap
            if step <= 0:
                step = 1
            start += step

        return chunks
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/dinoceli/Developer/tessera/repos/graphrag-core && uv run pytest tests/test_ingestion/test_chunker.py -v`

Expected: All 8 tests PASS.

- [ ] **Step 5: Run full test suite**

Run: `cd /Users/dinoceli/Developer/tessera/repos/graphrag-core && uv run pytest tests/ -v`

Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/graphrag_core/ingestion/chunker.py tests/test_ingestion/test_chunker.py
git commit -m "feat: add TokenChunker implementation"
```

---

### Task 6: IngestionPipeline

**Files:**
- Modify: `src/graphrag_core/ingestion/pipeline.py`
- Create: `tests/test_ingestion/test_pipeline.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_ingestion/test_pipeline.py`:

```python
"""Tests for IngestionPipeline."""

import pytest

from graphrag_core.models import (
    ChunkConfig,
    DocumentMetadata,
    ParsedDocument,
    TextSection,
)


class FakeParser:
    """Fake DocumentParser that returns a fixed document."""

    def __init__(self, doc: ParsedDocument):
        self._doc = doc

    async def parse(self, source: bytes, content_type: str) -> ParsedDocument:
        return self._doc


class FakeChunker:
    """Fake Chunker that returns one chunk per section."""

    def chunk(self, doc, config):
        from graphrag_core.models import DocumentChunk
        return [
            DocumentChunk(id=f"chunk-{i}", text=s.text, position=i)
            for i, s in enumerate(doc.sections)
        ]


class FakeEmbeddingModel:
    """Fake EmbeddingModel that returns fixed-length vectors."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 2.0, 3.0] for _ in texts]


def _make_doc() -> ParsedDocument:
    return ParsedDocument(
        sections=[
            TextSection(heading=None, text="Hello world."),
            TextSection(heading=None, text="Goodbye world."),
        ],
        metadata=DocumentMetadata(
            title="", source="", doc_type="text/plain",
            date=None, quarter=None, sha256="abc123",
        ),
    )


class TestIngestionPipeline:
    @pytest.mark.asyncio
    async def test_ingest_without_embedding(self):
        from graphrag_core.ingestion.pipeline import IngestionPipeline

        doc = _make_doc()
        pipeline = IngestionPipeline(
            parser=FakeParser(doc),
            chunker=FakeChunker(),
        )
        chunks = await pipeline.ingest(b"dummy", "text/plain")

        assert len(chunks) == 2
        assert chunks[0].text == "Hello world."
        assert chunks[1].text == "Goodbye world."
        assert chunks[0].embedding is None
        assert chunks[1].embedding is None

    @pytest.mark.asyncio
    async def test_ingest_with_embedding(self):
        from graphrag_core.ingestion.pipeline import IngestionPipeline

        doc = _make_doc()
        pipeline = IngestionPipeline(
            parser=FakeParser(doc),
            chunker=FakeChunker(),
            embedding_model=FakeEmbeddingModel(),
        )
        chunks = await pipeline.ingest(b"dummy", "text/plain")

        assert len(chunks) == 2
        assert chunks[0].embedding == [1.0, 2.0, 3.0]
        assert chunks[1].embedding == [1.0, 2.0, 3.0]

    @pytest.mark.asyncio
    async def test_ingest_uses_default_config(self):
        from graphrag_core.ingestion.pipeline import IngestionPipeline

        doc = _make_doc()

        class ConfigCapturingChunker:
            captured_config = None

            def chunk(self, doc, config):
                ConfigCapturingChunker.captured_config = config
                return []

        pipeline = IngestionPipeline(
            parser=FakeParser(doc),
            chunker=ConfigCapturingChunker(),
        )
        await pipeline.ingest(b"dummy", "text/plain")

        assert ConfigCapturingChunker.captured_config is not None
        assert ConfigCapturingChunker.captured_config.max_tokens == 512
        assert ConfigCapturingChunker.captured_config.overlap == 50

    @pytest.mark.asyncio
    async def test_ingest_uses_custom_config(self):
        from graphrag_core.ingestion.pipeline import IngestionPipeline

        doc = _make_doc()

        class ConfigCapturingChunker:
            captured_config = None

            def chunk(self, doc, config):
                ConfigCapturingChunker.captured_config = config
                return []

        pipeline = IngestionPipeline(
            parser=FakeParser(doc),
            chunker=ConfigCapturingChunker(),
        )
        custom = ChunkConfig(max_tokens=100, overlap=10)
        await pipeline.ingest(b"dummy", "text/plain", config=custom)

        assert ConfigCapturingChunker.captured_config.max_tokens == 100
        assert ConfigCapturingChunker.captured_config.overlap == 10

    @pytest.mark.asyncio
    async def test_pipeline_satisfies_protocol(self):
        from graphrag_core.ingestion.pipeline import IngestionPipeline as PipelineImpl
        from graphrag_core.interfaces import IngestionPipeline as PipelineProtocol

        assert isinstance(
            PipelineImpl(parser=FakeParser(_make_doc()), chunker=FakeChunker()),
            PipelineProtocol,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/dinoceli/Developer/tessera/repos/graphrag-core && uv run pytest tests/test_ingestion/test_pipeline.py -v`

Expected: FAIL — `ImportError: cannot import name 'IngestionPipeline' from 'graphrag_core.ingestion.pipeline'`

- [ ] **Step 3: Implement IngestionPipeline**

Replace contents of `src/graphrag_core/ingestion/pipeline.py`:

```python
"""BB1: Ingestion pipeline orchestrator."""

from __future__ import annotations

from graphrag_core.interfaces import Chunker, DocumentParser, EmbeddingModel
from graphrag_core.models import ChunkConfig, DocumentChunk


class IngestionPipeline:
    """Wires together parser, chunker, and optional embedding model."""

    def __init__(
        self,
        parser: DocumentParser,
        chunker: Chunker,
        embedding_model: EmbeddingModel | None = None,
    ) -> None:
        self._parser = parser
        self._chunker = chunker
        self._embedding_model = embedding_model

    async def ingest(
        self,
        source: bytes,
        content_type: str,
        config: ChunkConfig | None = None,
    ) -> list[DocumentChunk]:
        parsed = await self._parser.parse(source, content_type)
        chunks = self._chunker.chunk(parsed, config or ChunkConfig())

        if self._embedding_model is not None:
            embeddings = await self._embedding_model.embed([c.text for c in chunks])
            for chunk, emb in zip(chunks, embeddings):
                chunk.embedding = emb

        return chunks
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/dinoceli/Developer/tessera/repos/graphrag-core && uv run pytest tests/test_ingestion/test_pipeline.py -v`

Expected: All 5 tests PASS.

- [ ] **Step 5: Run full test suite**

Run: `cd /Users/dinoceli/Developer/tessera/repos/graphrag-core && uv run pytest tests/ -v`

Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/graphrag_core/ingestion/pipeline.py tests/test_ingestion/test_pipeline.py
git commit -m "feat: add IngestionPipeline implementation"
```

---

### Task 7: Clean up ingestion __init__.py and update top-level re-exports

**Files:**
- Modify: `src/graphrag_core/ingestion/__init__.py`
- Modify: `src/graphrag_core/__init__.py:5-37`

- [ ] **Step 1: Verify ingestion __init__.py imports work**

Run: `cd /Users/dinoceli/Developer/tessera/repos/graphrag-core && uv run python -c "from graphrag_core.ingestion import PdfParser, DocxParser, TextParser, MarkdownParser, TokenChunker, IngestionPipeline; print('OK')"`

Expected: `OK` (the __init__.py was already created with all imports in Task 3).

- [ ] **Step 2: Add ingestion implementations to top-level __init__.py**

In `src/graphrag_core/__init__.py`, add ingestion imports:

```python
from graphrag_core.ingestion import (
    DocxParser,
    IngestionPipeline,
    MarkdownParser,
    PdfParser,
    TextParser,
    TokenChunker,
)
```

Note: The `IngestionPipeline` here is the concrete class from `ingestion.pipeline`, which shadows the Protocol import from `interfaces`. Rename the Protocol import to avoid collision:

Actually, the better approach: do NOT re-export the concrete `IngestionPipeline` class from the top level — the Protocol `IngestionPipeline` is already exported. Users import the concrete class from `graphrag_core.ingestion` if they need it. Re-export only the parsers and chunker:

```python
from graphrag_core.ingestion import (
    DocxParser,
    MarkdownParser,
    PdfParser,
    TextParser,
    TokenChunker,
)
```

Add these to `__all__` as well:

```python
__all__ = [
    ...
    "DocxParser",
    "MarkdownParser",
    "PdfParser",
    "TextParser",
    "TokenChunker",
]
```

- [ ] **Step 3: Verify top-level imports work**

Run: `cd /Users/dinoceli/Developer/tessera/repos/graphrag-core && uv run python -c "from graphrag_core import PdfParser, DocxParser, TextParser, MarkdownParser, TokenChunker, IngestionPipeline; print('OK')"`

Expected: `OK`

- [ ] **Step 4: Run full test suite**

Run: `cd /Users/dinoceli/Developer/tessera/repos/graphrag-core && uv run pytest tests/ -v`

Expected: All tests pass.

- [ ] **Step 5: Delete the temporary protocol test file**

Remove `tests/test_ingestion/test_pipeline_protocol.py` — its assertion is now covered by `test_pipeline.py::test_pipeline_satisfies_protocol`.

- [ ] **Step 6: Run full test suite one final time**

Run: `cd /Users/dinoceli/Developer/tessera/repos/graphrag-core && uv run pytest tests/ -v`

Expected: All tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/graphrag_core/__init__.py tests/test_ingestion/
git commit -m "feat: add BB1 ingestion re-exports to public API"
```
