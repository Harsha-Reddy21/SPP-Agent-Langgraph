"""
streamlit_app.py — Streamlit chatbot frontend for the Q&A Chat Agent
Run with: streamlit run streamlit_app.py
"""

import os
import time
import tempfile
from copy import deepcopy
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

from models import AgentInput, AgentOutput, QAPair, FieldType, Category, QualityScore
from graph import run_agent, stream_agent

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SPP Agent — Product Profile Chat",
    page_icon="🤖",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .stChatMessage { max-width: 85%; }
    .qa-card {
        padding: 0.5rem 0.8rem;
        margin: 0.25rem 0;
        border-radius: 8px;
        border-left: 4px solid;
        font-size: 0.85rem;
    }
    .qa-answered { border-left-color: #28a745; background: #f0fff4; }
    .qa-pending  { border-left-color: #ffc107; background: #fffdf0; }
    .qa-skipped  { border-left-color: #6c757d; background: #f8f9fa; }
    .quality-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .grade-excellent { background: #d4edda; color: #155724; }
    .grade-good      { background: #d1ecf1; color: #0c5460; }
    .grade-fair      { background: #fff3cd; color: #856404; }
    .grade-poor      { background: #f8d7da; color: #721c24; }
    .node-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
        margin-right: 4px;
    }
    .node-router     { background: #e3f2fd; color: #1565c0; }
    .node-extractor  { background: #e8f5e9; color: #2e7d32; }
    .node-educator   { background: #fff3e0; color: #e65100; }
    .node-completion { background: #f3e5f5; color: #6a1b9a; }
    .node-document   { background: #fce4ec; color: #c62828; }
    .node-assembler  { background: #eceff1; color: #37474f; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SAMPLE Q&A (same fixture as server.py)
# ─────────────────────────────────────────────────────────────────────────────

def make_qa_list() -> list[QAPair]:
    return [
        QAPair(
            id="qa-1", question="What is the product title?",
            answer=None, field_type=FieldType.TEXT,
            category=Category.SOLUTION, required=True, sort_order=1,
            help_text="The name of your product or initiative.",
        ),
        QAPair(
            id="qa-2", question="What problem does this product solve?",
            answer=None, field_type=FieldType.TEXTAREA,
            category=Category.SOLUTION, required=True, sort_order=2,
            help_text="Describe the core problem in 2-3 sentences.",
        ),
        QAPair(
            id="qa-3", question="Who are the primary users?",
            answer=None, field_type=FieldType.TEXT,
            category=Category.USER, required=True, sort_order=1,
        ),
        QAPair(
            id="qa-4", question="What is the deployment region?",
            answer=None, field_type=FieldType.DROPDOWN,
            options=["US East", "US West", "EU", "Asia Pacific"],
            category=Category.TECHNICAL, required=True, sort_order=1,
        ),
        QAPair(
            id="qa-5", question="What is the preferred AI model?",
            answer=None, field_type=FieldType.DROPDOWN,
            options=["GPT-4", "Claude", "Gemini", "Llama"],
            category=Category.TECHNICAL, required=True, sort_order=2,
        ),
        QAPair(
            id="qa-6", question="Who is the development vendor/partner?",
            answer=None, field_type=FieldType.TEXT,
            category=Category.TECHNICAL, required=False, sort_order=3,
        ),
        QAPair(
            id="qa-7", question="What is the expected budget range?",
            answer=None, field_type=FieldType.DROPDOWN,
            options=["< $100K", "$100K-$500K", "$500K-$1M", "> $1M"],
            category=Category.GENERAL, required=False, sort_order=1,
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def apply_answers(qa_list: list[QAPair], extracted: list) -> list[QAPair]:
    qa_list = deepcopy(qa_list)
    answer_map = {e.question_id: e.answer for e in extracted}
    for qa in qa_list:
        if qa.id in answer_map:
            qa.answer = answer_map[qa.id]
    return qa_list


def read_uploaded_file(uploaded_file) -> str:
    """Read content from a Streamlit UploadedFile object."""
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix == ".pdf":
        from PyPDF2 import PdfReader
        import io
        reader = PdfReader(io.BytesIO(uploaded_file.read()))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()

    elif suffix == ".docx":
        import docx
        import io
        doc = docx.Document(io.BytesIO(uploaded_file.read()))
        return "\n".join(para.text for para in doc.paragraphs).strip()

    else:
        return uploaded_file.read().decode("utf-8").strip()


NODE_LABELS = {
    "router":            ("🔀 Router", "node-router"),
    "extractor":         ("📝 Extractor", "node-extractor"),
    "educator":          ("📚 Educator", "node-educator"),
    "completion":        ("✅ Completion", "node-completion"),
    "file_extractor":    ("📄 File Extractor", "node-document"),
    "document_uploader": ("📄 Document Upload", "node-document"),
    "assembler":         ("📦 Assembler", "node-assembler"),
}


def grade_css_class(grade: str) -> str:
    return f"grade-{grade.lower()}" if grade.lower() in ("excellent", "good", "fair", "poor") else "grade-fair"


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────────────

if "qa" not in st.session_state:
    st.session_state.qa = make_qa_list()
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


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Q&A Status + File Upload
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📋 Product Profile")
    st.caption(f"Profile ID: `{st.session_state.profile_id}`")

    # ── Quality score ─────────────────────────────────────────────────────
    if st.session_state.quality:
        qs = st.session_state.quality
        css = grade_css_class(qs.grade)
        st.markdown(
            f'<div class="quality-badge {css}">'
            f'{qs.grade} — {qs.overall:.0f}%</div>',
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        col1.metric("Required", f"{qs.required_completion:.0f}%")
        col2.metric("Optional", f"{qs.optional_completion:.0f}%")

        if qs.suggestions:
            with st.expander("💡 Suggestions"):
                for s in qs.suggestions:
                    st.write(f"- {s}")
    else:
        st.info("Quality score will appear after your first message.")

    st.divider()

    # ── Q&A cards ─────────────────────────────────────────────────────────
    st.markdown("### Questions")
    for q in st.session_state.qa:
        if q.answer is not None:
            css_class = "qa-answered"
            icon = "✅"
            detail = f"**{q.answer}**"
        elif q.skipped:
            css_class = "qa-skipped"
            icon = "⏭️"
            detail = "*Skipped*"
        else:
            css_class = "qa-pending"
            icon = "⏳"
            detail = "*Pending*"

        req = "🔴" if q.required else "⚪"
        st.markdown(
            f'<div class="qa-card {css_class}">'
            f'{icon} {req} <b>{q.question}</b><br/>{detail}</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Document Upload ───────────────────────────────────────────────────
    st.markdown("### 📎 Upload Document")
    uploaded_file = st.file_uploader(
        "Upload a document to auto-fill the profile",
        type=["txt", "pdf", "docx", "md", "csv"],
        key="file_uploader",
    )
    if uploaded_file:
        try:
            content = read_uploaded_file(uploaded_file)
            st.session_state.doc_content = content
            st.session_state.doc_name = uploaded_file.name
            st.success(f"📄 **{uploaded_file.name}** loaded ({len(content):,} chars)")
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            st.session_state.doc_content = None
            st.session_state.doc_name = None

    # ── Reset button ──────────────────────────────────────────────────────
    st.divider()
    if st.button("🔄 Reset Profile", use_container_width=True):
        st.session_state.qa = make_qa_list()
        st.session_state.messages = []
        st.session_state.history = []
        st.session_state.quality = None
        st.session_state.doc_content = None
        st.session_state.doc_name = None
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CHAT AREA
# ─────────────────────────────────────────────────────────────────────────────

st.title("🤖 SPP Agent — Product Profile Chat")

if not os.getenv("OPENAI_API_KEY"):
    st.error("⚠️ `OPENAI_API_KEY` not set. Add it to your `.env` file.")
    st.stop()

# ── Render chat history ──────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍💼" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])
        if msg.get("extracted"):
            with st.expander("📝 Extracted Answers"):
                for ea in msg["extracted"]:
                    st.write(f"- **{ea['qid']}**: {ea['answer']}")
        if msg.get("interactive"):
            ie = msg["interactive"]
            st.info(f"💬 **{ie['type']}** for question `{ie['qid']}`: {', '.join(ie['options'])}")


# ─────────────────────────────────────────────────────────────────────────────
# CHAT INPUT + AGENT EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

user_input = st.chat_input("Type your message or describe what the uploaded document contains...")

if user_input:
    # ── Resolve document context ──────────────────────────────────────────
    doc_content = st.session_state.doc_content
    doc_name = st.session_state.doc_name
    upload_prompt = None

    if doc_content:
        upload_prompt = user_input
        display_msg = f"📄 *[Document: {doc_name}]*\n\n{user_input}"
    else:
        display_msg = user_input

    # ── Show user message ─────────────────────────────────────────────────
    st.session_state.messages.append({"role": "user", "content": display_msg})
    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(display_msg)

    # ── Build agent input ─────────────────────────────────────────────────
    agent_input = AgentInput(
        product_profile_id=st.session_state.profile_id,
        current_qa=deepcopy(st.session_state.qa),
        user_message=user_input,
        conversation_history=st.session_state.history,
        uploaded_document_content=doc_content,
        upload_prompt=upload_prompt,
    )

    # ── Stream agent execution ────────────────────────────────────────────
    with st.chat_message("assistant", avatar="🤖"):
        status_container = st.status("🔄 Processing...", expanded=True)

        final_message = ""
        extracted_answers = []
        interactive_el = None
        quality_score = None
        all_answered = False

        with status_container:
            for event in stream_agent(agent_input):
                node_name = event["node"]
                node_state = event["state"]

                label, css = NODE_LABELS.get(node_name, (node_name, "node-assembler"))
                st.markdown(
                    f'<span class="node-badge {css}">{label}</span> completed',
                    unsafe_allow_html=True,
                )

                # Capture state from processing nodes
                if "agent_message" in node_state and node_state["agent_message"]:
                    final_message = node_state["agent_message"]
                if "extracted_answers" in node_state:
                    extracted_answers = node_state["extracted_answers"]
                if "interactive_elements" in node_state and node_state["interactive_elements"]:
                    interactive_el = node_state["interactive_elements"]
                if "quality_score" in node_state and node_state["quality_score"]:
                    quality_score = node_state["quality_score"]
                if "all_questions_answered" in node_state:
                    all_answered = node_state["all_questions_answered"]

            status_container.update(label="✅ Done", state="complete", expanded=False)

        # ── Stream the agent message character by character ───────────────
        if final_message:
            placeholder = st.empty()
            streamed = ""
            for char in final_message:
                streamed += char
                placeholder.markdown(streamed + "▌")
                time.sleep(0.01)
            placeholder.markdown(streamed)

        # ── Show extracted answers ────────────────────────────────────────
        extracted_display = []
        if extracted_answers:
            with st.expander("📝 Extracted Answers", expanded=True):
                for ea in extracted_answers:
                    qid = ea.question_id if hasattr(ea, "question_id") else ea.get("question_id", "")
                    ans = ea.answer if hasattr(ea, "answer") else ea.get("answer", "")
                    st.write(f"- **{qid}**: {ans}")
                    extracted_display.append({"qid": qid, "answer": ans})

        # ── Show interactive element ──────────────────────────────────────
        interactive_display = None
        if interactive_el:
            ie_type = interactive_el.type if hasattr(interactive_el, "type") else interactive_el.get("type", "")
            ie_qid = interactive_el.question_id if hasattr(interactive_el, "question_id") else interactive_el.get("question_id", "")
            ie_opts = interactive_el.options if hasattr(interactive_el, "options") else interactive_el.get("options", [])
            st.info(f"💬 **{ie_type}** for `{ie_qid}`: {', '.join(ie_opts)}")
            interactive_display = {"type": ie_type, "qid": ie_qid, "options": ie_opts}

        # ── Show completion banner ────────────────────────────────────────
        if all_answered:
            st.success("🎉 **All questions answered!** Review the profile in the sidebar.")

    # ── Update session state ──────────────────────────────────────────────
    if extracted_answers:
        st.session_state.qa = apply_answers(st.session_state.qa, extracted_answers)

    if quality_score:
        if isinstance(quality_score, dict):
            quality_score = QualityScore(**quality_score)
        st.session_state.quality = quality_score

    st.session_state.history.append({"role": "user", "content": user_input})
    st.session_state.history.append({"role": "assistant", "content": final_message})

    # Store message for history rendering
    assistant_msg = {"role": "assistant", "content": final_message}
    if extracted_display:
        assistant_msg["extracted"] = extracted_display
    if interactive_display:
        assistant_msg["interactive"] = interactive_display
    st.session_state.messages.append(assistant_msg)

    # Clear document after processing
    if doc_content:
        st.session_state.doc_content = None
        st.session_state.doc_name = None

    st.rerun()
