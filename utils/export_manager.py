"""Export generated results as TXT, DOCX or PDF files.

The three builders return bytes so Streamlit's download buttons can serve the
file directly without writing temporary files.
"""

from __future__ import annotations

import io
import re
from datetime import date
from typing import Mapping, Optional

import pymupdf
from docx import Document
from docx.shared import Pt, RGBColor

# Characters that do not exist in the built-in PDF font (Helvetica/Latin-1).
_PDF_REPLACEMENTS = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u2022": "-",
    "\u2026": "...",
    "\u00a0": " ",
    "\u2192": "->",
}

PAGE_WIDTH = 595  # A4 at 72 dpi
PAGE_HEIGHT = 842
MARGIN = 56


def slugify(title: str, fallback: str = "contentcraft") -> str:
    """Create a safe file name from a title."""
    value = re.sub(r"[^\w\s-]", "", (title or "").strip(), flags=re.UNICODE)
    value = re.sub(r"[\s_-]+", "-", value).strip("-").lower()
    return (value[:60].strip("-") or fallback)


def export_filename(title: str, extension: str) -> str:
    """Return a timestamped file name for downloads."""
    return f"{slugify(title)}-{date.today().isoformat()}.{extension.lstrip('.')}"


def _meta_lines(meta: Optional[Mapping[str, str]]) -> list[str]:
    if not meta:
        return []
    return [f"{key}: {value}" for key, value in meta.items() if value not in (None, "")]


def build_txt(title: str, body: str, meta: Optional[Mapping[str, str]] = None) -> bytes:
    """Create a plain-text export."""
    title = (title or "ContentCraft Result").strip()
    lines = [title, "=" * min(len(title), 70), ""]
    lines.extend(_meta_lines(meta))
    if meta:
        lines.append("")
    lines.append((body or "").strip())
    lines.append("")
    lines.append("Generated with ContentCraft (local AI via Ollama).")
    return "\n".join(lines).encode("utf-8")


def _add_markdown_runs(paragraph, text: str) -> None:
    """Add **bold** segments to a DOCX paragraph without a full Markdown parser."""
    for index, part in enumerate(re.split(r"\*\*(.+?)\*\*", text)):
        if not part:
            continue
        run = paragraph.add_run(part)
        run.bold = index % 2 == 1


def build_docx(title: str, body: str, meta: Optional[Mapping[str, str]] = None) -> bytes:
    """Create a basic formatted Word document."""
    document = Document()
    document.core_properties.title = (title or "ContentCraft Result").strip()

    heading = document.add_heading((title or "ContentCraft Result").strip(), level=0)
    for run in heading.runs:
        run.font.color.rgb = RGBColor(31, 41, 55)

    for line in _meta_lines(meta):
        paragraph = document.add_paragraph()
        run = paragraph.add_run(line)
        run.italic = True
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(100, 116, 139)

    if meta:
        document.add_paragraph()

    for raw_line in (body or "").splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            document.add_paragraph()
            continue

        if line.startswith("### "):
            document.add_heading(line[4:].strip(), level=3)
        elif line.startswith("## "):
            document.add_heading(line[3:].strip(), level=2)
        elif line.startswith("# "):
            document.add_heading(line[2:].strip(), level=1)
        elif re.match(r"^\s*[-*+]\s+", line):
            text = re.sub(r"^\s*[-*+]\s+", "", line)
            _add_markdown_runs(document.add_paragraph(style="List Bullet"), text)
        elif re.match(r"^\s*\d+[.)]\s+", line):
            text = re.sub(r"^\s*\d+[.)]\s+", "", line)
            _add_markdown_runs(document.add_paragraph(style="List Number"), text)
        else:
            _add_markdown_runs(document.add_paragraph(), line)

    paragraph = document.add_paragraph()
    run = paragraph.add_run("Generated with ContentCraft - local AI via Ollama.")
    run.italic = True
    run.font.size = Pt(8)
    run.font.color.rgb = RGBColor(148, 163, 184)

    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def _pdf_safe(text: str) -> str:
    for source, replacement in _PDF_REPLACEMENTS.items():
        text = text.replace(source, replacement)
    return text.encode("latin-1", "replace").decode("latin-1")


def build_pdf(title: str, body: str, meta: Optional[Mapping[str, str]] = None) -> bytes:
    """Create a simple multi-page PDF using PyMuPDF's built-in font."""
    document = pymupdf.open()
    state = {"page": document.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT), "y": MARGIN}

    left = MARGIN
    right = PAGE_WIDTH - MARGIN
    max_width = right - left

    def new_page() -> None:
        state["page"] = document.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
        state["y"] = MARGIN

    def write(
        text: str = "",
        fontsize: int = 11,
        color: tuple[float, float, float] = (0.1, 0.12, 0.16),
        before: int = 0,
        after: int = 2,
        width_ratio: float = 1.0,
    ) -> None:
        nonlocal max_width
        state["y"] += before
        if not text:
            state["y"] += after
            return

        # Helvetica average character width is about 0.52 * font size.
        chars = max(20, int((max_width * width_ratio) / (fontsize * 0.52)))
        import textwrap

        wrapped = textwrap.wrap(
            _pdf_safe(text),
            width=chars,
            break_long_words=True,
            break_on_hyphens=True,
            replace_whitespace=True,
        ) or [""]

        leading = fontsize + 5
        for line in wrapped:
            if state["y"] + leading > PAGE_HEIGHT - MARGIN:
                new_page()
            state["page"].insert_text(
                (left, state["y"] + fontsize),
                line,
                fontsize=fontsize,
                fontname="helv",
                color=color,
            )
            state["y"] += leading
        state["y"] += after

    write(title or "ContentCraft Result", fontsize=20, color=(0.1, 0.15, 0.3), after=4)
    state["page"].draw_line(
        (left, state["y"]), (right, state["y"]), color=(0.8, 0.82, 0.9), width=1
    )
    state["y"] += 10

    for line in _meta_lines(meta):
        write(line, fontsize=9, color=(0.4, 0.45, 0.55), after=1)
    if meta:
        state["y"] += 8

    for raw_line in (body or "").splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            state["y"] += 6
        elif line.startswith("### "):
            write(line[4:].strip(), fontsize=12, color=(0.15, 0.2, 0.4), before=8)
        elif line.startswith("## "):
            write(line[3:].strip(), fontsize=14, color=(0.12, 0.17, 0.35), before=10)
        elif line.startswith("# "):
            write(line[2:].strip(), fontsize=16, color=(0.1, 0.15, 0.3), before=12)
        else:
            write(line, fontsize=11)

    write("Generated with ContentCraft - local AI via Ollama.", fontsize=8, color=(0.55, 0.6, 0.7), before=16)
    return document.tobytes()
