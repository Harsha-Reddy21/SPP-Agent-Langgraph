"""
streamlit_app.py — Streamlit chatbot frontend for the Q&A Chat Agent
Connects to the FastAPI backend (api.py) for all agent operations.
Run with: streamlit run streamlit_app.py
"""

import os
import io
import json
import time
from copy import deepcopy
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="SPP Agent — Product Profile Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — dark polished theme
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* ── Global ─────────────────────────────────────────────── */
    .block-container { padding-top: 1.5rem; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #13161B 0%, #1A1D23 100%);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #C8CDD3; }

    /* ── Header ─────────────────────────────────────────────── */
    .app-header {
        background: linear-gradient(135deg, #6C63FF 0%, #4F46E5 50%, #7C3AED 100%);
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }
    .app-header h1 {
        margin: 0; font-size: 1.5rem; color: #fff; font-weight: 700;
    }
    .app-header p {
        margin: 0; font-size: 0.85rem; color: rgba(255,255,255,0.8);
    }

    /* ── Q&A Cards ──────────────────────────────────────────── */
    .qa-card {
        padding: 0.6rem 0.9rem;
        margin: 0.3rem 0;
        border-radius: 10px;
        border-left: 4px solid;
        font-size: 0.82rem;
        transition: transform 0.15s;
    }
    .qa-card:hover { transform: translateX(3px); }
    .qa-answered {
        border-left-color: #22C55E;
        background: rgba(34, 197, 94, 0.08);
    }
    .qa-answered b { color: #4ADE80; }
    .qa-pending {
        border-left-color: #F59E0B;
        background: rgba(245, 158, 11, 0.06);
    }
    .qa-pending b { color: #FBBF24; }
    .qa-skipped {
        border-left-color: #64748B;
        background: rgba(100, 116, 139, 0.06);
    }
    .qa-skipped b { color: #94A3B8; }

    /* ── Quality badge ──────────────────────────────────────── */
    .quality-badge {
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 0.02em;
    }
    .grade-excellent { background: rgba(34,197,94,0.15); color: #4ADE80; border: 1px solid rgba(34,197,94,0.3); }
    .grade-good      { background: rgba(59,130,246,0.15); color: #60A5FA; border: 1px solid rgba(59,130,246,0.3); }
    .grade-fair      { background: rgba(245,158,11,0.15); color: #FBBF24; border: 1px solid rgba(245,158,11,0.3); }
    .grade-poor      { background: rgba(239,68,68,0.15);  color: #F87171; border: 1px solid rgba(239,68,68,0.3); }

    /* ── Progress bar ───────────────────────────────────────── */
    .progress-container {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        height: 8px;
        margin: 0.5rem 0 1rem 0;
        overflow: hidden;
    }
    .progress-bar {
        height: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, #6C63FF, #22C55E);
        transition: width 0.5s ease;
    }

    /* ── Node streaming badges ──────────────────────────────── */
    .node-pill {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 3px 10px;
        border-radius: 14px;
        font-size: 0.72rem;
        font-weight: 600;
        margin: 2px 3px;
    }
    .node-router     { background: rgba(99,102,241,0.15); color: #818CF8; }
    .node-extractor  { background: rgba(34,197,94,0.15);  color: #4ADE80; }
    .node-educator   { background: rgba(251,191,36,0.15); color: #FBBF24; }
    .node-completion { background: rgba(168,85,247,0.15); color: #C084FC; }
    .node-document   { background: rgba(236,72,153,0.15); color: #F472B6; }
    .node-assembler  { background: rgba(148,163,184,0.12); color: #94A3B8; }

    /* ── Extracted answer pills ─────────────────────────────── */
    .extract-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(34,197,94,0.1);
        border: 1px solid rgba(34,197,94,0.2);
        border-radius: 8px;
        padding: 4px 10px;
        margin: 3px 0;
        font-size: 0.8rem;
    }
    .extract-pill .qid { color: #6C63FF; font-weight: 600; }
    .extract-pill .ans { color: #C8CDD3; }

    /* ── Sidebar section titles ─────────────────────────────── */
    .sidebar-title {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748B;
        margin: 0.8rem 0 0.4rem 0;
    }

    /* ── Upload area ────────────────────────────────────────── */
    .upload-zone {
        border: 2px dashed rgba(108,99,255,0.3);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        background: rgba(108,99,255,0.04);
        transition: border-color 0.2s;
    }
    .upload-zone:hover { border-color: rgba(108,99,255,0.6); }

    /* ── Chat bubbles ───────────────────────────────────────── */
    [data-testid="stChatMessage"] {
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.04) !important;
    }

    /* ── Metrics ────────────────────────────────────────────── */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.03);
        border-radius: 10px;
        padding: 0.6rem;
        border: 1px solid rgba(255,255,255,0.06);
    }
    [data-testid="stMetricValue"] { font-size: 1.3rem !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

NODE_LABELS = {
    "router":            ("🔀 Router", "node-router"),
    "extractor":         ("📝 Extractor", "node-extractor"),
    "educator":          ("📚 Educator", "node-educator"),
    "completion":        ("✅ Completion", "node-completion"),
    "file_extractor":    ("📄 File Extractor", "node-document"),
    "document_uploader": ("📄 Document Upload", "node-document"),
    "assembler":         ("📦 Assembler", "node-assembler"),
}


def grade_css(grade: str) -> str:
    g = grade.lower()
    return f"grade-{g}" if g in ("excellent", "good", "fair", "poor") else "grade-fair"


def read_uploaded_file(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".pdf":
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(uploaded_file.read()))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    elif suffix == ".docx":
        import docx
        doc = docx.Document(io.BytesIO(uploaded_file.read()))
        return "\n".join(para.text for para in doc.paragraphs).strip()
    else:
        return uploaded_file.read().decode("utf-8").strip()


def call_agent_stream(payload: dict):
    """Call /chat/stream SSE endpoint with streaming."""
    resp = requests.post(
        f"{API_BASE}/chat/stream",
        json=payload,
        stream=True,
        timeout=120,
    )
    resp.raise_for_status()
    for line in resp.iter_lines(decode_unicode=True):
        if line and line.startswith("data: "):
            data = json.loads(line[6:])
            yield data


def call_agent(payload: dict) -> dict:
    """Call /chat non-streaming endpoint."""
    resp = requests.post(f"{API_BASE}/chat", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def apply_answers(qa_list: list[dict], extracted: list[dict]) -> list[dict]:
    qa_list = deepcopy(qa_list)
    answer_map = {ea["question_id"]: ea["answer"] for ea in extracted}
    for qa in qa_list:
        if qa["id"] in answer_map:
            qa["answer"] = answer_map[qa["id"]]
    return qa_list


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

def init_qa():
    """Fetch Q&A template from backend or use fallback."""
    try:
        resp = requests.get(f"{API_BASE}/qa-template", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return []

if "qa" not in st.session_state:
    st.session_state.qa = init_qa()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []
if "profile_id" not in st.session_state:
    st.session_state.profile_id = "profile-ui-001"
if "quality" not in st.session_state:
    st.session_state.quality = None
if "doc_content" not in st.session_state:
    st.session_state.doc_content = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None
if "pending_widget" not in st.session_state:
    st.session_state.pending_widget = None
if "widget_answer" not in st.session_state:
    st.session_state.widget_answer = None


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    # ── Logo / Brand ──────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding: 0.5rem 0 0.3rem 0;">
        <span style="font-size:2rem;">🤖</span>
        <h2 style="margin:0; font-size:1.2rem; color:#C8CDD3;">SPP Agent</h2>
        <span style="font-size:0.7rem; color:#64748B;">Smart Product Profile</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<p style="text-align:center; font-size:0.7rem; color:#475569;">Profile: <code>{st.session_state.profile_id}</code></p>', unsafe_allow_html=True)

    # ── Quality Score ─────────────────────────────────────────────────────
    if st.session_state.quality:
        qs = st.session_state.quality
        grade = qs.get("grade", "Fair")
        overall = qs.get("overall", 0)
        css = grade_css(grade)

        st.markdown(f'<div style="text-align:center; margin: 0.5rem 0;"><span class="quality-badge {css}">{grade} — {overall:.0f}%</span></div>', unsafe_allow_html=True)

        # Progress bar
        st.markdown(f'''
        <div class="progress-container">
            <div class="progress-bar" style="width: {overall}%"></div>
        </div>
        ''', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        col1.metric("Required", f"{qs.get('required_completion', 0):.0f}%")
        col2.metric("Optional", f"{qs.get('optional_completion', 0):.0f}%")

        suggestions = qs.get("suggestions", [])
        if suggestions:
            with st.expander("💡 Suggestions", expanded=False):
                for s in suggestions:
                    st.caption(f"→ {s}")

    st.markdown('<div class="sidebar-title">📋 Questions</div>', unsafe_allow_html=True)

    # ── Q&A Cards ─────────────────────────────────────────────────────────
    answered_count = sum(1 for q in st.session_state.qa if q.get("answer") is not None)
    total_count = len(st.session_state.qa)
    st.caption(f"{answered_count} / {total_count} answered")

    for q in st.session_state.qa:
        answer = q.get("answer")
        skipped = q.get("skipped", False)
        required = q.get("required", True)

        if answer is not None:
            css_class = "qa-answered"
            icon = "✅"
            detail = f"<span style='color:#4ADE80'>{answer}</span>"
        elif skipped:
            css_class = "qa-skipped"
            icon = "⏭️"
            detail = "<span style='color:#64748B'>Skipped</span>"
        else:
            css_class = "qa-pending"
            icon = "⏳"
            detail = "<span style='color:#64748B'>Awaiting answer…</span>"

        req = '<span style="color:#EF4444; font-size:0.6rem;">●</span>' if required else '<span style="color:#475569; font-size:0.6rem;">○</span>'

        st.markdown(
            f'<div class="qa-card {css_class}">'
            f'{icon} {req} <b>{q["question"]}</b><br/>{detail}</div>',
            unsafe_allow_html=True,
        )

    # ── Document Upload ───────────────────────────────────────────────────
    st.markdown('<div class="sidebar-title">📎 Upload Document</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop a file to auto-fill answers",
        type=["txt", "pdf", "docx", "md", "csv"],
        key="file_uploader",
        label_visibility="collapsed",
    )
    if uploaded_file:
        try:
            content = read_uploaded_file(uploaded_file)
            st.session_state.doc_content = content
            st.session_state.doc_name = uploaded_file.name
            st.success(f"📄 {uploaded_file.name} — {len(content):,} chars")
        except Exception as e:
            st.error(f"Parse error: {e}")
            st.session_state.doc_content = None
            st.session_state.doc_name = None

    # ── Reset ─────────────────────────────────────────────────────────────
    st.markdown("")
    if st.button("🔄 Reset Profile", use_container_width=True, type="secondary"):
        st.session_state.qa = init_qa()
        st.session_state.messages = []
        st.session_state.history = []
        st.session_state.quality = None
        st.session_state.doc_content = None
        st.session_state.doc_name = None
        st.session_state.pending_widget = None
        st.session_state.widget_answer = None
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CHAT AREA
# ─────────────────────────────────────────────────────────────────────────────

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div>
        <h1>🤖 Product Profile Agent</h1>
        <p>Chat to fill your product profile or upload a document to auto-extract answers</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Backend health check ─────────────────────────────────────────────────────
try:
    health = requests.get(f"{API_BASE}/health", timeout=3)
    if health.status_code != 200:
        st.error(f"⚠️ Backend not reachable at `{API_BASE}`. Start it with `python api.py`")
        st.stop()
except requests.ConnectionError:
    st.error(f"⚠️ Cannot connect to backend at `{API_BASE}`. Start it with:\n```\npython api.py\n```")
    st.stop()

# ── Welcome message ──────────────────────────────────────────────────────────
if not st.session_state.messages:
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(
            "👋 **Welcome!** I'm your Product Profile assistant.\n\n"
            "You can:\n"
            "- **Chat** with me to answer profile questions one by one\n"
            "- **Upload a document** (sidebar) and tell me what to extract\n\n"
            "Let's start — **What is the product title?**"
        )

# ── Render history ────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    avatar = "🧑‍💼" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("extracted"):
            with st.expander("📝 Extracted Answers", expanded=False):
                for ea in msg["extracted"]:
                    st.markdown(
                        f'<div class="extract-pill"><span class="qid">{ea["qid"]}</span>'
                        f'<span class="ans">{ea["answer"]}</span></div>',
                        unsafe_allow_html=True,
                    )


# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE WIDGET — for dropdown / radio / multi_select questions
# ─────────────────────────────────────────────────────────────────────────────

def _find_next_constrained_question() -> dict | None:
    """Find the next unanswered question that has options (dropdown/radio/multi_select)."""
    for q in st.session_state.qa:
        if q.get("answer") is not None or q.get("skipped", False):
            continue
        if q.get("field_type") in ("dropdown", "radio", "multi_select") and q.get("options"):
            return q
    return None


def _render_interactive_widget():
    """Render a selection widget for the current pending constrained question."""
    widget = st.session_state.pending_widget
    if not widget:
        return

    qid = widget.get("id", "")
    question = widget.get("question", "")
    field_type = widget.get("field_type", "dropdown")
    options = widget.get("options", [])

    st.markdown(
        f'<div style="background: rgba(108,99,255,0.08); border: 1px solid rgba(108,99,255,0.2); '
        f'border-radius: 12px; padding: 1rem; margin: 0.5rem 0;">'
        f'<p style="color:#818CF8; font-weight:600; margin:0 0 0.5rem 0;">'
        f'💬 {question}</p></div>',
        unsafe_allow_html=True,
    )

    widget_key = f"widget_{qid}"

    if field_type == "multi_select":
        selected = st.multiselect(
            "Select all that apply:",
            options=options,
            key=widget_key,
            label_visibility="collapsed",
        )
    elif field_type == "radio":
        selected = st.radio(
            "Choose one:",
            options=options,
            key=widget_key,
            index=None,
            label_visibility="collapsed",
            horizontal=len(options) <= 4,
        )
    else:  # dropdown
        selected = st.selectbox(
            "Select one:",
            options=["— Select —"] + options,
            key=widget_key,
            label_visibility="collapsed",
        )
        if selected == "— Select —":
            selected = None

    col1, col2 = st.columns([1, 4])
    with col1:
        submit = st.button("✅ Submit", key=f"submit_{qid}", use_container_width=True, type="primary")
    with col2:
        skip = st.button("⏭️ Skip", key=f"skip_{qid}", use_container_width=True)

    if submit and selected:
        if isinstance(selected, list):
            answer_text = ", ".join(selected)
        else:
            answer_text = selected
        st.session_state.pending_widget = None
        st.session_state.widget_answer = answer_text
        st.rerun()

    if skip:
        st.session_state.pending_widget = None
        st.session_state.widget_answer = None
        st.rerun()


# Render widget if one is pending
if st.session_state.pending_widget:
    _render_interactive_widget()


# ─────────────────────────────────────────────────────────────────────────────
# CHAT INPUT
# ─────────────────────────────────────────────────────────────────────────────

user_input = st.chat_input("Type your answer or ask a question…")

# ── Handle widget selection as input ──────────────────────────────────────────
if "widget_answer" in st.session_state and st.session_state.widget_answer:
    user_input = st.session_state.widget_answer
    st.session_state.widget_answer = None

if user_input:
    doc_content = st.session_state.doc_content
    doc_name = st.session_state.doc_name
    upload_prompt = None

    if doc_content:
        upload_prompt = user_input
        display_msg = f"📄 *[{doc_name}]*\n\n{user_input}"
    else:
        display_msg = user_input

    # Show user message
    st.session_state.messages.append({"role": "user", "content": display_msg})
    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(display_msg)

    # Build payload for API
    payload = {
        "product_profile_id": st.session_state.profile_id,
        "current_qa": st.session_state.qa,
        "user_message": user_input,
        "conversation_history": st.session_state.history,
        "uploaded_document_content": doc_content,
        "upload_prompt": upload_prompt,
    }

    # ── Streaming agent call ──────────────────────────────────────────────
    with st.chat_message("assistant", avatar="🤖"):
        status_container = st.status("🔄 Processing…", expanded=True)

        final_result = None

        try:
            with status_container:
                for event in call_agent_stream(payload):
                    if event["type"] == "node":
                        node = event["node"]
                        label, css = NODE_LABELS.get(node, (node, "node-assembler"))
                        st.markdown(
                            f'<span class="node-pill {css}">{label}</span> ✓',
                            unsafe_allow_html=True,
                        )
                    elif event["type"] == "result":
                        final_result = event

                status_container.update(label="✅ Complete", state="complete", expanded=False)

        except requests.ConnectionError:
            status_container.update(label="❌ Connection failed", state="error", expanded=False)
            st.error("Lost connection to the backend.")
            st.stop()
        except Exception as e:
            status_container.update(label="❌ Error", state="error", expanded=False)
            st.error(f"Agent error: {e}")
            # Fallback: try non-streaming
            try:
                final_result = call_agent(payload)
            except Exception:
                st.stop()

        if final_result:
            agent_msg = final_result.get("agent_message", "")
            extracted = final_result.get("extracted_answers", [])
            quality = final_result.get("quality_score")
            all_done = final_result.get("all_questions_answered", False)

            # Stream the message text
            if agent_msg:
                placeholder = st.empty()
                streamed = ""
                for char in agent_msg:
                    streamed += char
                    placeholder.markdown(streamed + "▌")
                    time.sleep(0.008)
                placeholder.markdown(streamed)

            # Show extracted answers
            extracted_display = []
            if extracted:
                with st.expander("📝 Extracted Answers", expanded=True):
                    for ea in extracted:
                        qid = ea.get("question_id", "")
                        ans = ea.get("answer", "")
                        st.markdown(
                            f'<div class="extract-pill"><span class="qid">{qid}</span>'
                            f'<span class="ans">{ans}</span></div>',
                            unsafe_allow_html=True,
                        )
                        extracted_display.append({"qid": qid, "answer": ans})

            # Completion banner
            if all_done:
                st.success("🎉 **All questions answered!** Review the profile in the sidebar.")

            # ── Detect next constrained question for widget ───────────────
            ie = final_result.get("interactive_elements")
            if ie and ie.get("question_id") and ie.get("options"):
                # Use interactive element from agent response
                matching_q = None
                for q in st.session_state.qa:
                    if q["id"] == ie["question_id"]:
                        matching_q = q
                        break
                if matching_q and matching_q.get("answer") is None:
                    st.session_state.pending_widget = matching_q
            else:
                # Auto-detect next constrained unanswered question
                next_constrained = _find_next_constrained_question()
                if next_constrained:
                    st.session_state.pending_widget = next_constrained
                else:
                    st.session_state.pending_widget = None

            # ── Update session state ──────────────────────────────────────
            if extracted:
                st.session_state.qa = apply_answers(st.session_state.qa, extracted)

            if quality:
                st.session_state.quality = quality

            st.session_state.history.append({"role": "user", "content": user_input})
            st.session_state.history.append({"role": "assistant", "content": agent_msg})

            assistant_msg = {"role": "assistant", "content": agent_msg}
            if extracted_display:
                assistant_msg["extracted"] = extracted_display
            st.session_state.messages.append(assistant_msg)

            # Clear document after use
            if doc_content:
                st.session_state.doc_content = None
                st.session_state.doc_name = None

            st.rerun()

