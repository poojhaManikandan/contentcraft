"""ContentCraft - AI content generation and text summarization system.

Run with:
    streamlit run app.py

The app supports a hosted OpenAI-compatible API for public deployment and an
optional local Ollama provider for development.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from ai_engine.errors import ModelError
from ai_engine.local_model import ModelStatus
from ai_engine.model_client import TextModel
from ai_engine.provider import create_model, get_provider_config
from modules.content_generator import (
    CONTENT_TYPES,
    MAX_WORDS,
    MIN_WORDS,
    TONES,
    generate_content,
)
from modules.document_processor import (
    DocumentProcessingError,
    extract_document,
    process_document,
)
from modules.keyword_extractor import extract_keywords, keywords_as_text
from modules.summarizer import METHODS, SUMMARY_LENGTHS, summarize_text
from modules.text_rewriter import MAX_REWRITE_CHARS, REWRITE_STYLES, rewrite_text
from utils.export_manager import build_docx, build_pdf, build_txt, export_filename, slugify
from utils.text_cleaner import compression_ratio, word_count

PAGES = (
    "Dashboard",
    "Content Generator",
    "Text Summarizer",
    "Document Summarizer",
    "Text Rewriter",
    "Keyword Extractor",
)

CSS_PATH = Path(__file__).parent / "assets" / "styles.css"

st.set_page_config(
    page_title="ContentCraft",
    page_icon=":material/edit_note:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------------------------------------------------
# Cached resources and status helpers
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_model() -> TextModel:
    """One shared model client for the whole session."""
    return create_model()


@st.cache_data(ttl=10, show_spinner=False)
def load_status() -> ModelStatus:
    """Check the configured provider at most every 10 seconds."""
    return create_model().check()


@st.cache_data(show_spinner=False, max_entries=20)
def extract_document_cached(filename: str, data: bytes) -> str:
    """Extract document text once per uploaded file."""
    return extract_document(filename, data)


def load_css() -> None:
    if CSS_PATH.exists():
        st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def require_model(status: ModelStatus) -> bool:
    """Show a clear error and return False when AI generation is unavailable."""
    if status.connected and status.model_available:
        return True

    config = get_provider_config()
    st.error(status.message)
    if config.hosted:
        st.markdown(
            '<div class="cc-note">ContentCraft will not show fake AI results. Add '
            "AI_API_KEY to Streamlit secrets, then refresh the model status.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="cc-note">ContentCraft will not show fake AI results. '
            "Start Ollama locally, then refresh the model status.</div>",
            unsafe_allow_html=True,
        )
        if status.connected and not status.model_available:
            st.code(f"ollama pull {status.model}", language="bash")
    return False


# ----------------------------------------------------------------------
# Shared UI components
# ----------------------------------------------------------------------
def page_head(eyebrow: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="cc-page-head">
            <p class="cc-eyebrow">{html.escape(eyebrow)}</p>
            <h1>{html.escape(title)}</h1>
            <p>{html.escape(description)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(title: str, description: str = "") -> None:
    st.markdown(
        f"""
        <div class="cc-section-title">
            <h2>{html.escape(title)}</h2>
            <p>{html.escape(description)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_copy_button(text: str, key: str) -> None:
    """A real clipboard button for the localhost/HTTPS Streamlit app."""
    widget_id = "cc-copy-" + hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
    payload = json.dumps(text).replace("</", "<\\/")
    components.html(
        f"""
        <style>
            html, body {{ margin: 0; padding: 0; background: transparent; }}
            body {{ font-family: "Segoe UI", Arial, sans-serif; }}
            .cc-copy-btn {{
                width: 100%; height: 38px; border-radius: 10px; cursor: pointer;
                border: 1px solid #d8deea; background: #ffffff; color: #0f172a;
                font-size: 13px; font-weight: 600; font-family: inherit;
            }}
            .cc-copy-btn:hover {{ border-color: #a5b4fc; color: #4338ca; background: #f8faff; }}
            .cc-copy-btn.ok {{ border-color: #a7f3d0; background: #ecfdf5; color: #047857; }}
        </style>
        <button id="{widget_id}" class="cc-copy-btn" type="button">Copy to clipboard</button>
        <script>
        (function () {{
            const button = document.getElementById("{widget_id}");
            const value = {payload};
            function markCopied() {{
                button.textContent = "Copied";
                button.classList.add("ok");
                setTimeout(function () {{
                    button.textContent = "Copy to clipboard";
                    button.classList.remove("ok");
                }}, 1800);
            }}
            button.addEventListener("click", async function () {{
                try {{
                    await navigator.clipboard.writeText(value);
                    markCopied();
                }} catch (error) {{
                    const area = document.createElement("textarea");
                    area.value = value;
                    area.style.position = "fixed";
                    area.style.opacity = "0";
                    document.body.appendChild(area);
                    area.focus();
                    area.select();
                    try {{
                        document.execCommand("copy");
                        markCopied();
                    }} catch (fallbackError) {{
                        button.textContent = "Copy unavailable - select the text instead";
                    }}
                    document.body.removeChild(area);
                }}
            }});
        }})();
        </script>
        """,
        height=46,
        scrolling=False,
    )


def render_copy_and_exports(
    title: str,
    body: str,
    key_prefix: str,
    meta: dict | None = None,
) -> None:
    """Render the copy control and the three download formats."""
    st.markdown("###### Copy and download")
    copy_column, download_column = st.columns([1.15, 2.85], vertical_alignment="center")

    with copy_column:
        render_copy_button(body, key_prefix)

    with download_column:
        builders = (
            ("TXT", "txt", build_txt, "text/plain"),
            ("DOCX", "docx", build_docx, "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
            ("PDF", "pdf", build_pdf, "application/pdf"),
        )
        columns = st.columns(3)
        for column, (label, extension, builder, mime) in zip(columns, builders):
            try:
                data = builder(title, body, meta)
            except Exception as exc:  # A broken export must not hide the result.
                column.button(label, disabled=True, width="stretch", key=f"{key_prefix}-{extension}-error")
                column.caption(f"Export failed: {exc}")
                continue
            column.download_button(
                label,
                data=data,
                file_name=export_filename(title, extension),
                mime=mime,
                width="stretch",
                key=f"{key_prefix}-{extension}",
            )


def render_result(
    title: str,
    content: str,
    key_prefix: str,
    meta: dict | None = None,
    badges: list[str] | None = None,
) -> None:
    """Display a generated result in a clean bordered card with exports."""
    section_title("Result")
    with st.container(border=True):
        if badges:
            badge_html = "".join(
                f'<span class="cc-badge primary">{html.escape(str(badge))}</span>'
                for badge in badges
            )
            st.markdown(f'<div class="cc-result-meta">{badge_html}</div>', unsafe_allow_html=True)
        st.markdown(content)
    render_copy_and_exports(title, content, key_prefix, meta)


def show_stats(items: list[tuple[str, str]]) -> None:
    html_items = "".join(
        f'<div class="cc-stat"><span class="label">{html.escape(label)}</span>'
        f'<span class="value">{html.escape(str(value))}</span></div>'
        for label, value in items
    )
    st.markdown(f'<div class="cc-stats">{html_items}</div>', unsafe_allow_html=True)


def now_label() -> str:
    return datetime.now().strftime("%d %b %Y, %H:%M")


# ----------------------------------------------------------------------
# Sidebar and dashboard
# ----------------------------------------------------------------------
def render_sidebar(status: ModelStatus) -> str:
    config = get_provider_config()
    with st.sidebar:
        st.markdown(
            """
            <div class="cc-brand">
                <div class="cc-brand-mark">CC</div>
                <div>
                    <div class="cc-brand-name">ContentCraft</div>
                    <div class="cc-brand-sub">AI writing studio</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        page = st.radio("Navigation", PAGES, label_visibility="collapsed", key="navigation")

        if status.model_available:
            dot, label = "ok", "AI model ready" if config.hosted else "Local model ready"
        elif status.connected:
            dot, label = "warn", "AI model unavailable" if config.hosted else "Model unavailable"
        else:
            dot, label = "off", "AI not configured" if config.hosted else "Ollama offline"

        body = (
            "AI-powered writing"
            if config.hosted
            else f"{html.escape(status.model)}<br>{html.escape(status.host)}"
        )
        st.markdown(
            f"""
            <div class="cc-side-status">
                <div class="cc-side-status-head"><span class="cc-dot {dot}"></span>{html.escape(label)}</div>
                <div class="cc-side-status-body">{body}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Refresh AI status", width="stretch", key="sidebar_refresh"):
            load_status.clear()
            st.rerun()

        st.markdown("---")
        st.caption("ContentCraft v1.0")
        st.caption("Public deployment ready" if config.hosted else "Local development mode")
    return page


def page_dashboard(status: ModelStatus) -> None:
    config = get_provider_config()
    kicker = "AI writing - Public demo" if config.hosted else "Local AI - Privacy-first"
    description = (
        "ContentCraft is a writing assistant for blogs, essays, reports, emails and study "
        "notes. Summarize long documents, rewrite text in a new style and pull out key terms "
        "with a secure AI model."
        if config.hosted
        else "ContentCraft is a self-contained writing assistant for blogs, essays, reports, "
        "emails and study notes. Summarize long documents, rewrite text in a new style and "
        "pull out key terms with a model running on your own computer."
    )
    st.markdown(
        f"""
        <div class="cc-hero">
            <div class="cc-hero-kicker">{html.escape(kicker)}</div>
            <h1>Write, summarize and study smarter.</h1>
            <p>{html.escape(description)}</p>
            <div class="cc-chips">
                <span>AI-powered</span>
                <span>Six writing tools</span>
                <span>No account needed</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_title("What you can do", "Six focused tools, no account and no database.")
    features = [
        ("01", "Content Generator", "Create well-structured blogs, articles, essays, emails, reports, social posts and study notes."),
        ("02", "Text Summarizer", "Compare an offline TF-IDF extractive summary with an abstractive summary from the AI model."),
        ("03", "Document Summarizer", "Upload a PDF, DOCX or TXT file. Long files are chunked and summarized section by section."),
        ("04", "Text Rewriter", "Rewrite a passage as professional, academic, formal, simple or concise while keeping its meaning."),
        ("05", "Keyword Extractor", "Find important terms, TF-IDF scores and the most informative sentences without calling the LLM."),
        ("06", "Export Center", "Copy any result or download it as a formatted TXT, Word document or PDF."),
    ]
    cards = "".join(
        f'<div class="cc-card"><div class="cc-num">{number}</div><h3>{title}</h3><p>{description}</p></div>'
        for number, title, description in features
    )
    st.markdown(f'<div class="cc-grid">{cards}</div>', unsafe_allow_html=True)

    section_title("How the application works", "Four simple steps from an idea or document to a finished result.")
    run_text = (
        "The AI service generates text when you run an AI tool. Extractive tools run without the AI model."
        if config.hosted
        else "Ollama serves the local model on this machine. Extractive tools run without the AI model."
    )
    steps = [
        ("Step 01", "Provide input", "Type a topic or paste text, or upload a PDF, DOCX or TXT file."),
        ("Step 02", "Choose options", "Pick a content type, tone, summary length, rewrite style or keyword count."),
        ("Step 03", "Run AI", run_text),
        ("Step 04", "Review and export", "Read the result, copy it, or download it as TXT, DOCX or PDF."),
    ]
    step_html = "".join(
        f'<div class="cc-step"><div class="n">{label}</div><h4>{title}</h4><p>{text}</p></div>'
        for label, title, text in steps
    )
    st.markdown(f'<div class="cc-steps">{step_html}</div>', unsafe_allow_html=True)

    section_title("AI model status", "AI text is generated only when the configured model is available.")
    if status.model_available:
        dot, title = "ok", "AI model ready" if config.hosted else "Local model ready"
    elif status.connected:
        dot, title = "warn", "AI model unavailable" if config.hosted else "Model missing"
    else:
        dot, title = "off", "AI not configured" if config.hosted else "Ollama is offline"

    status_meta = ""
    if not config.hosted:
        installed = (
            ", ".join(status.available_models[:6]) if status.available_models else "None detected"
        )
        status_meta = (
            '<div class="cc-status-meta">'
            f"<span><strong>Model</strong> {html.escape(status.model)}</span>"
            f"<span><strong>Installed</strong> {html.escape(installed)}</span>"
            f"<span><strong>Endpoint</strong> {html.escape(status.host)}</span>"
            "</div>"
        )

    st.markdown(
        f"""
        <div class="cc-status-card">
            <div class="cc-status-title"><span class="cc-dot {dot}"></span>{html.escape(title)}</div>
            <div class="cc-status-detail">{html.escape(status.message)}</div>
            {status_meta}
        </div>
        """,
        unsafe_allow_html=True,
    )

    action_column, note_column = st.columns([1, 3], vertical_alignment="center")
    with action_column:
        if st.button("Refresh status", width="stretch", key="dashboard_refresh"):
            load_status.clear()
            st.rerun()
    with note_column:
        note = (
            "The AI service is contacted only when you run an AI action. "
            "Generated text should always be reviewed before use."
            if config.hosted
            else "The first request after starting Ollama can take longer while the model loads. "
            "Generated text should always be reviewed before use."
        )
        st.caption(note)

    st.markdown(
        '<div class="cc-note" style="margin-top:18px;">Extractive summarization and keyword extraction '
        "work without the AI model. Content generation, abstractive summaries and rewriting require "
        "a configured AI provider.</div>",
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------
# Module pages
# ----------------------------------------------------------------------
def page_content_generator(status: ModelStatus) -> None:
    page_head(
        "Module 01",
        "Content Generator",
        "Create a structured first draft with the AI model. Choose the format, tone and target length.",
    )

    existing = st.session_state.get("content_result")
    input_column, tip_column = st.columns([1.55, 1])

    with input_column:
        topic = st.text_input(
            "Topic",
            placeholder="e.g. The impact of renewable energy on rural economies",
            max_chars=300,
            key="cg_topic",
        )
        row_one, row_two = st.columns(2)
        content_type = row_one.selectbox("Content type", CONTENT_TYPES, key="cg_type")
        tone = row_two.selectbox("Writing tone", TONES, key="cg_tone")
        word_count_value = st.slider(
            "Target word count",
            MIN_WORDS,
            MAX_WORDS,
            300,
            step=50,
            key="cg_words",
        )
        extra_instructions = st.text_input(
            "Additional instructions (optional)",
            placeholder="e.g. Include three practical examples",
            max_chars=300,
            key="cg_extra",
        )

    with tip_column:
        st.markdown(
            """
            <div class="cc-card">
                <div class="cc-num">TIP</div>
                <h3>Better topics, better drafts</h3>
                <p>
                    A specific topic produces a more useful result. Instead of "climate change",
                    try "three ways cities can reduce summer heat".
                </p>
                <p style="margin-top:10px;">
                    The model writes only original text and is instructed not to invent
                    statistics or sources. Always check facts before submitting work.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    button_one, button_two = st.columns(2)
    generate_clicked = button_one.button(
        "Generate content", type="primary", width="stretch", key="cg_generate"
    )
    regenerate_clicked = button_two.button(
        "Regenerate",
        width="stretch",
        disabled=existing is None,
        key="cg_regenerate",
        help="Uses the settings from the last generated draft.",
    )

    if generate_clicked:
        if require_model(status):
            try:
                with st.spinner(f"Writing your {content_type.lower()}..."):
                    content = generate_content(
                        topic=topic,
                        content_type=content_type,
                        tone=tone,
                        word_count=int(word_count_value),
                        extra_instructions=extra_instructions,
                        variation=0,
                        model=get_model(),
                    )
                st.session_state["content_result"] = {
                    "topic": topic.strip(),
                    "content_type": content_type,
                    "tone": tone,
                    "target_words": int(word_count_value),
                    "extra": extra_instructions.strip(),
                    "content": content,
                    "variation": 0,
                    "created": now_label(),
                }
            except (ValueError, ModelError) as exc:
                st.error(str(exc))

    if regenerate_clicked and existing:
        if require_model(status):
            try:
                variation = int(existing.get("variation", 0)) + 1
                with st.spinner("Writing a fresh version..."):
                    content = generate_content(
                        topic=existing["topic"],
                        content_type=existing["content_type"],
                        tone=existing["tone"],
                        word_count=int(existing["target_words"]),
                        extra_instructions=existing.get("extra", ""),
                        variation=variation,
                        model=get_model(),
                    )
                existing = dict(existing)
                existing["content"] = content
                existing["variation"] = variation
                existing["created"] = now_label()
                st.session_state["content_result"] = existing
            except (ValueError, ModelError) as exc:
                st.error(str(exc))

    result = st.session_state.get("content_result")
    if result:
        meta = {
            "Content type": result["content_type"],
            "Tone": result["tone"],
            "Target length": f"{result['target_words']} words",
            "Generated": result["created"],
        }
        render_result(
            title=f"{result['topic']} - {result['content_type']}",
            content=result["content"],
            key_prefix=f"content-{result['variation']}",
            meta=meta,
            badges=[
                result["content_type"],
                result["tone"],
                f"{word_count(result['content'])} words",
                f"Version {result['variation'] + 1}",
            ],
        )
    else:
        st.caption("Your generated draft will appear here.")


def page_text_summarizer(status: ModelStatus) -> None:
    page_head(
        "Module 02",
        "Text Summarizer",
        "Paste an article, notes or a report. Compare an offline TF-IDF extractive summary with an abstractive AI summary.",
    )

    text = st.text_area(
        "Text to summarize",
        height=280,
        placeholder="Paste at least a few paragraphs here...",
        key="summary_text",
    )
    st.caption(f"{word_count(text)} words in the input")

    control_one, control_two = st.columns(2)
    method = control_one.radio("Summarization method", METHODS, horizontal=True, key="summary_method")
    length = control_two.selectbox("Summary length", SUMMARY_LENGTHS, index=1, key="summary_length")

    if method == "Extractive":
        st.caption("Extractive mode uses TF-IDF sentence scoring and works without the AI model.")
    else:
        st.caption("Abstractive mode uses the AI model to write a new summary.")

    if st.button("Summarize text", type="primary", width="stretch", key="summary_run"):
        model_ready = method == "Extractive" or require_model(status)
        if not model_ready:
            pass
        elif not text.strip():
            st.error("Please paste some text before summarizing.")
        else:
            progress = None
            callback = None
            if method == "Abstractive":
                progress = st.progress(0.0, text="Preparing the AI model...")

                def callback(fraction: float, message: str) -> None:
                    progress.progress(fraction, text=message)

            try:
                with st.spinner("Summarizing..."):
                    result = summarize_text(
                        text=text,
                        method=method,
                        length=length,
                        model=get_model() if method == "Abstractive" else None,
                        progress_callback=callback,
                    )
                summary = result["summary"]
                st.session_state["summary_result"] = {
                    "original": text,
                    "summary": summary,
                    "method": method,
                    "length": length,
                    "chunk_count": result["chunk_count"],
                    "original_words": word_count(text),
                    "summary_words": word_count(summary),
                    "created": now_label(),
                }
            except (ValueError, ModelError) as exc:
                st.error(str(exc))
            finally:
                if progress is not None:
                    progress.empty()

    result = st.session_state.get("summary_result")
    if result:
        ratio = compression_ratio(result["original_words"], result["summary_words"])
        show_stats(
            [
                ("Original words", str(result["original_words"])),
                ("Summary words", str(result["summary_words"])),
                ("Compression", f"{ratio}% of original"),
                ("Sections", str(result["chunk_count"] or 1)),
            ]
        )
        meta = {
            "Method": result["method"],
            "Length": result["length"],
            "Original words": str(result["original_words"]),
            "Summary words": str(result["summary_words"]),
            "Generated": result["created"],
        }
        render_result(
            title=f"Text summary ({result['length']})",
            content=result["summary"],
            key_prefix=f"summary-{result['method']}-{result['length']}",
            meta=meta,
            badges=[result["method"], result["length"], f"{result['summary_words']} words"],
        )
        with st.expander("View the original text"):
            st.text_area(
                "Original text",
                value=result["original"],
                height=220,
                disabled=True,
                key="summary_original_view",
            )
    else:
        st.caption("Your summary will appear here.")


def page_document_summarizer(status: ModelStatus) -> None:
    page_head(
        "Module 03",
        "Document Summarizer",
        "Upload a PDF, DOCX or TXT file. Long documents are split into chunks, summarized and combined into one result.",
    )

    uploaded = st.file_uploader(
        "Upload a document",
        type=["pdf", "docx", "txt"],
        help="Maximum file size: 25 MB. Scanned image-only PDFs are not supported.",
        key="doc_upload",
    )

    data = None
    extracted_text = None
    extraction_error = None

    if uploaded is not None:
        data = uploaded.getvalue()
        size_kb = len(data) / 1024
        st.markdown(
            f"""
            <div class="cc-stat" style="max-width:520px;">
                <span class="label">Selected file</span>
                <span class="value" style="font-size:0.92rem;">{html.escape(uploaded.name)}</span>
                <span class="label" style="margin-top:4px;">{size_kb:,.1f} KB - {html.escape(Path(uploaded.name).suffix.upper())}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        try:
            extracted_text = extract_document_cached(uploaded.name, data)
            st.success(f"Readable text found: {word_count(extracted_text):,} words.")
            with st.expander("Preview extracted text"):
                st.text_area(
                    "Extracted text",
                    value=extracted_text[:12000],
                    height=220,
                    disabled=True,
                    key="doc_preview",
                )
                if len(extracted_text) > 12000:
                    st.caption("Preview limited to the first 12,000 characters.")
        except (DocumentProcessingError, ValueError) as exc:
            extraction_error = str(exc)
            st.error(extraction_error)

    control_one, control_two = st.columns(2)
    method = control_one.radio("Summarization method", METHODS, horizontal=True, key="doc_method")
    length = control_two.selectbox("Summary length", SUMMARY_LENGTHS, index=1, key="doc_length")

    if method == "Extractive":
        st.caption("Extractive mode uses TF-IDF and works without the AI model.")
    else:
        st.caption("Abstractive mode summarizes each chunk with the AI model, then combines the results.")

    summarize_clicked = st.button(
        "Summarize document",
        type="primary",
        width="stretch",
        key="doc_run",
        disabled=uploaded is None or extracted_text is None,
    )

    if summarize_clicked and uploaded is not None and data is not None:
        model_ready = method == "Extractive" or require_model(status)
        if model_ready:
            progress = st.progress(0.0, text="Preparing the document...")

            def callback(fraction: float, message: str) -> None:
                progress.progress(fraction, text=message)

            try:
                result = process_document(
                    filename=uploaded.name,
                    data=data,
                    length=length,
                    method=method,
                    model=get_model() if method == "Abstractive" else None,
                    progress_callback=callback,
                )
                result["file_id"] = hashlib.sha1(data).hexdigest()
                result["created"] = now_label()
                st.session_state["document_result"] = result
            except (ValueError, ModelError, DocumentProcessingError) as exc:
                st.error(str(exc))
            finally:
                progress.empty()

    result = st.session_state.get("document_result")
    file_id = hashlib.sha1(data).hexdigest() if data else None
    if result and file_id and result.get("file_id") == file_id:
        show_stats(
            [
                ("Document", result["filename"]),
                ("Extracted words", f"{result['word_count']:,}"),
                ("Chunks", str(result["chunk_count"])),
                ("Summary words", f"{result['summary_word_count']:,}"),
            ]
        )

        points = result.get("key_points", [])
        points_body = "\n".join(f"- {point}" for point in points) if points else "No key points were extracted."
        export_body = f"{result['summary']}\n\n## Important points\n\n{points_body}"
        meta = {
            "Source document": result["filename"],
            "Extracted words": f"{result['word_count']:,}",
            "Method": result["method"],
            "Length": result["length"],
            "Generated": result["created"],
        }

        section_title("Document result", f"{result['filename']} - {result['method']} summary")
        summary_tab, points_tab = st.tabs(["Final summary", "Important points"])
        with summary_tab:
            with st.container(border=True):
                st.markdown(result["summary"])
        with points_tab:
            with st.container(border=True):
                if points:
                    for index, point in enumerate(points, start=1):
                        st.markdown(f"**{index}.** {point}")
                else:
                    st.caption("No important points were extracted from this document.")
            if result.get("key_points_error"):
                st.warning(f"Important points could not be generated: {result['key_points_error']}")

        render_copy_and_exports(
            title=f"Summary of {result['filename']}",
            body=export_body,
            key_prefix=f"document-{result['file_id'][:8]}",
            meta=meta,
        )
    elif uploaded is not None and extraction_error is None:
        st.caption("Your document summary will appear here.")

    if not status.model_available:
        st.caption("Abstractive document summaries need the AI model. Extractive summaries still work offline.")


def page_text_rewriter(status: ModelStatus) -> None:
    page_head(
        "Module 04",
        "Text Rewriter",
        "Paste a paragraph and rewrite it in a different style. The prompt tells the model to preserve facts and meaning.",
    )

    text = st.text_area(
        "Original text",
        height=250,
        placeholder="Paste the text you want to rewrite...",
        key="rewrite_text",
    )
    count = len(text)
    st.caption(f"{word_count(text)} words - {count:,} / {MAX_REWRITE_CHARS:,} characters")

    style_column, note_column = st.columns([1, 2])
    style = style_column.selectbox("Rewrite style", REWRITE_STYLES, key="rewrite_style")
    note_column.markdown(
        '<div class="cc-note">The rewrite uses the AI model. It will not add outside facts, '
        "but you should still read the result carefully before using it.</div>",
        unsafe_allow_html=True,
    )

    if st.button("Rewrite text", type="primary", width="stretch", key="rewrite_run"):
        if require_model(status):
            try:
                with st.spinner(f"Rewriting in a {style.lower()} style..."):
                    rewritten = rewrite_text(text=text, style=style, model=get_model())
                st.session_state["rewrite_result"] = {
                    "original": text,
                    "rewritten": rewritten,
                    "style": style,
                    "original_words": word_count(text),
                    "rewritten_words": word_count(rewritten),
                    "created": now_label(),
                }
            except (ValueError, ModelError) as exc:
                st.error(str(exc))

    result = st.session_state.get("rewrite_result")
    if result:
        show_stats(
            [
                ("Style", result["style"]),
                ("Original words", str(result["original_words"])),
                ("Rewritten words", str(result["rewritten_words"])),
                ("Generated", result["created"]),
            ]
        )
        meta = {
            "Style": result["style"],
            "Original words": str(result["original_words"]),
            "Rewritten words": str(result["rewritten_words"]),
            "Generated": result["created"],
        }
        render_result(
            title=f"Rewritten text ({result['style']})",
            content=result["rewritten"],
            key_prefix=f"rewrite-{slugify(result['style'])}",
            meta=meta,
            badges=[result["style"], f"{result['rewritten_words']} words"],
        )
        with st.expander("View the original text"):
            st.text_area(
                "Original",
                value=result["original"],
                height=200,
                disabled=True,
                key="rewrite_original_view",
            )
    else:
        st.caption("Your rewritten text will appear here.")


def page_keyword_extractor(status: ModelStatus) -> None:
    page_head(
        "Module 05",
        "Keyword Extractor",
        "Identify important terms and informative sentences with scikit-learn TF-IDF. This module does not call the language model.",
    )

    text = st.text_area(
        "Text for keyword analysis",
        height=270,
        placeholder="Paste an article, report or study notes...",
        key="keyword_text",
    )
    st.caption(f"{word_count(text)} words in the input")

    count_column, note_column = st.columns([1, 2])
    top_n = count_column.slider("Number of keywords", 5, 25, 12, key="keyword_count")
    note_column.markdown(
        '<div class="cc-note">Keywords are ranked by TF-IDF across sentences. '
        "A term that matters in several sentences ranks above a one-off mention.</div>",
        unsafe_allow_html=True,
    )

    if st.button("Extract keywords", type="primary", width="stretch", key="keyword_run"):
        if not text.strip():
            st.error("Please paste some text before extracting keywords.")
        else:
            try:
                result = extract_keywords(text, top_n=int(top_n))
                st.session_state["keyword_result"] = {
                    "text": text,
                    "result": result,
                    "top_n": int(top_n),
                    "created": now_label(),
                }
            except ValueError as exc:
                st.error(str(exc))

    stored = st.session_state.get("keyword_result")
    if stored:
        result = stored["result"]
        keywords = result["keywords"]
        show_stats(
            [
                ("Words analyzed", f"{result['word_count']:,}"),
                ("Sentences", str(result["sentence_count"])),
                ("Keywords", str(len(keywords))),
                ("Method", "TF-IDF"),
            ]
        )

        section_title("Top keywords", "TF-IDF score and a 0-1 weight for comparison.")
        table = pd.DataFrame(
            [
                {
                    "Rank": index,
                    "Keyword": item["keyword"],
                    "Weight": float(item["normalized_score"]),
                    "TF-IDF score": float(item["score"]),
                }
                for index, item in enumerate(keywords, start=1)
            ]
        )
        st.dataframe(
            table,
            hide_index=True,
            width="stretch",
            column_config={
                "Weight": st.column_config.ProgressColumn(
                    "Weight", min_value=0.0, max_value=1.0, format="%.2f"
                ),
                "TF-IDF score": st.column_config.NumberColumn(format="%.4f"),
            },
        )

        if len(table) > 1:
            chart_data = table.head(10).set_index("Keyword")["Weight"]
            st.bar_chart(chart_data, height=260, width="stretch")

        section_title("Important sentences", "Sentences with the highest combined TF-IDF weight.")
        with st.container(border=True):
            for index, item in enumerate(result["sentences"], start=1):
                st.markdown(f"**{index}.** {item['sentence']}")
                st.caption(f"TF-IDF score: {item['score']}")

        export_body = keywords_as_text(result)
        meta = {
            "Words analyzed": f"{result['word_count']:,}",
            "Sentences": str(result["sentence_count"]),
            "Keywords": str(len(keywords)),
            "Generated": stored["created"],
        }
        render_copy_and_exports(
            title="Keyword analysis",
            body=export_body,
            key_prefix="keywords",
            meta=meta,
        )
    else:
        st.caption("Keyword results will appear here.")


# ----------------------------------------------------------------------
# Application entry point
# ----------------------------------------------------------------------
def main() -> None:
    load_css()
    status = load_status()
    page = render_sidebar(status)

    if page == "Dashboard":
        page_dashboard(status)
    elif page == "Content Generator":
        page_content_generator(status)
    elif page == "Text Summarizer":
        page_text_summarizer(status)
    elif page == "Document Summarizer":
        page_document_summarizer(status)
    elif page == "Text Rewriter":
        page_text_rewriter(status)
    elif page == "Keyword Extractor":
        page_keyword_extractor(status)

if __name__ == "__main__":
    main()
