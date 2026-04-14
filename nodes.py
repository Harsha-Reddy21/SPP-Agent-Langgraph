"""
nodes.py — All LangGraph node functions for the Q&A Chat Agent
"""

import json
import logging
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from models import (
    GraphState, QAPair, ExtractedAnswer,
    InteractiveElement, Category,
)
from prompts import (
    ROUTER_SYSTEM, ROUTER_USER,
    EXTRACTOR_SYSTEM, EXTRACTOR_USER,
    EDUCATION_SYSTEM, EDUCATION_USER,
    COMPLETION_SYSTEM, COMPLETION_USER,
    FILE_EXTRACTION_SYSTEM, FILE_EXTRACTION_USER,
    DOCUMENT_UPLOAD_SYSTEM, DOCUMENT_UPLOAD_USER,
)
from quality import compute_quality_score

logger = logging.getLogger(__name__)


# ── LLM setup ────────────────────────────────────────────────────────────────

def _get_llm(temperature: float = 0.3) -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-4o",
        temperature=temperature,
        max_tokens=2048,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _qa_to_dict(qa: QAPair) -> dict:
    return {
        "id":         qa.id,
        "question":   qa.question,
        "answer":     qa.answer,
        "field_type": qa.field_type,
        "options":    qa.options,
        "category":   qa.category,
        "required":   qa.required,
        "sort_order": qa.sort_order,
        "skipped":    qa.skipped,
        "help_text":  qa.help_text,
    }


def _qa_summary(qa_pairs: list[QAPair]) -> str:
    """Short summary for the router — avoids sending full Q&A JSON."""
    answered   = [q for q in qa_pairs if q.answer is not None]
    unanswered = [q for q in qa_pairs if q.answer is None and not q.skipped]
    skipped    = [q for q in qa_pairs if q.skipped]
    return (
        f"Total: {len(qa_pairs)} | Answered: {len(answered)} | "
        f"Unanswered: {len(unanswered)} | Skipped: {len(skipped)}"
    )


def _format_history(history: list[dict], last_n: int = 6) -> str:
    if not history:
        return "No prior conversation."
    recent = history[-last_n:]
    lines = []
    for turn in recent:
        role = turn.get("role", "unknown").capitalize()
        lines.append(f"{role}: {turn.get('content', '')}")
    return "\n".join(lines)


def _call_llm(system: str, user: str, temperature: float = 0.3) -> dict:
    """Call the LLM and parse the JSON response."""
    logger.info("[LLM CALL] temperature=%.1f", temperature)
    logger.debug("[LLM SYSTEM PROMPT] %s", system[:200])
    logger.debug("[LLM USER PROMPT] %s", user[:300])
    llm = _get_llm(temperature)
    messages = [SystemMessage(content=system), HumanMessage(content=user)]
    response = llm.invoke(messages)
    raw = response.content.strip()
    logger.info("[LLM RAW RESPONSE] %s", raw[:500])

    # strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error("JSON parse error: %s\nRaw: %s", e, raw)
        raise


def _next_unanswered(qa_pairs: list[QAPair]) -> QAPair | None:
    """
    Return the next question to ask using priority rules:
    required first → category order → sort_order.
    Skipped questions are moved to the end.
    """
    category_order = {
        Category.SOLUTION:  0,
        Category.USER:      1,
        Category.TECHNICAL: 2,
        Category.GENERAL:   3,
    }

    unanswered = [q for q in qa_pairs if q.answer is None and not q.skipped]
    if not unanswered:
        # check skipped ones as fallback
        unanswered = [q for q in qa_pairs if q.answer is None and q.skipped]

    if not unanswered:
        return None

    return sorted(
        unanswered,
        key=lambda q: (
            0 if q.required else 1,
            category_order.get(q.category, 99),
            q.sort_order,
        )
    )[0]


def _apply_extracted_answers(
    qa_pairs: list[QAPair],
    extracted: list[ExtractedAnswer],
) -> list[QAPair]:
    """Merge extracted answers back into the Q&A list.

    For dropdown / radio / multi_select fields the answer is accepted ONLY
    when every value exists in the question's predefined options list
    (case-insensitive match, stored with the original casing from options).
    """
    answer_map = {e.question_id: e.answer for e in extracted}
    updated = []
    for qa in qa_pairs:
        if qa.id in answer_map:
            value = answer_map[qa.id]
            # Validate constrained fields
            if qa.options and qa.field_type in ("dropdown", "radio", "multi_select"):
                opts_lower = {o.lower(): o for o in qa.options}
                if qa.field_type == "multi_select" and isinstance(value, list):
                    matched = [opts_lower[v.lower()] for v in value if v.lower() in opts_lower]
                    value = matched if matched else None
                else:
                    value = opts_lower.get(str(value).lower())
                if value is None:
                    logger.warning(
                        "[APPLY] Rejected answer for %s — value not in options %s",
                        qa.id, qa.options,
                    )
            if value is not None:
                qa = qa.model_copy(update={"answer": value})
        updated.append(qa)
    return updated


# ─────────────────────────────────────────────────────────────────────────────
# NODE 1 — ROUTER
# ─────────────────────────────────────────────────────────────────────────────

def router_node(state: GraphState) -> GraphState:
    """Classify the user's intent to decide which node runs next."""
    logger.info("="*60)
    logger.info("[NODE: ROUTER] Starting intent classification")
    inp  = state.agent_input
    qa   = inp.current_qa
    logger.info("[ROUTER] User message: %s", inp.user_message)
    logger.info("[ROUTER] Q&A summary: %s", _qa_summary(qa))
    all_answered = all(q.answer is not None or q.skipped for q in qa)

    user_prompt = ROUTER_USER.format(
        qa_summary=_qa_summary(qa),
        has_file_extraction=inp.file_extraction_results is not None,
        all_answered=all_answered,
        user_message=inp.user_message,
    )

    try:
        result = _call_llm(ROUTER_SYSTEM, user_prompt, temperature=0.0)
        route  = result.get("route", "extract")
    except Exception:
        route = "extract"   # safe default

    # override with hard rules
    if all_answered:
        route = "complete"
        logger.info("[ROUTER] Hard override → complete (all answered)")
    elif inp.uploaded_document_content:
        route = "document_upload"
        logger.info("[ROUTER] Hard override → document_upload (document content present)")
    elif inp.file_extraction_results:
        route = "file_extraction"
        logger.info("[ROUTER] Hard override → file_extraction")

    logger.info("[ROUTER] Final route decision: %s", route)
    logger.info("="*60)
    return state.model_copy(update={"route": route})


# ─────────────────────────────────────────────────────────────────────────────
# NODE 2 — EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

def extractor_node(state: GraphState) -> GraphState:
    """Extract answers from the user message and formulate the next question."""
    logger.info("="*60)
    logger.info("[NODE: EXTRACTOR] Extracting answers from user message")
    inp     = state.agent_input
    history = _format_history(inp.conversation_history)

    # Pre-compute the next unanswered question BEFORE extraction
    pre_next = _next_unanswered(inp.current_qa)
    next_q_id = pre_next.id if pre_next else "NONE"
    next_q_text = pre_next.question if pre_next else "All questions answered."
    next_q_type = pre_next.field_type if pre_next else "N/A"
    next_q_options = json.dumps(pre_next.options) if pre_next and pre_next.options else "N/A"

    # Build a compact summary of already-answered questions (no full list)
    answered_summary = "\n".join(
        f"  {q.id}: {q.question} → {q.answer}"
        for q in inp.current_qa if q.answer is not None
    ) or "(none yet)"

    user_prompt = EXTRACTOR_USER.format(
        product_profile_id=inp.product_profile_id,
        answered_summary=answered_summary,
        next_question_id=next_q_id,
        next_question_text=next_q_text,
        next_question_type=next_q_type,
        next_question_options=next_q_options,
        history=history,
        user_message=inp.user_message,
    )

    result = _call_llm(EXTRACTOR_SYSTEM, user_prompt)

    # Only accept answers for the current next question — reject anything else
    extracted = []
    for e in result.get("extractedAnswers", []):
        if e.get("answer") is None:
            continue
        if e.get("questionId") != next_q_id:
            logger.warning(
                "[EXTRACTOR] Rejected answer for %s — only %s is being asked",
                e.get("questionId"), next_q_id,
            )
            continue
        extracted.append(ExtractedAnswer(question_id=e["questionId"], answer=e["answer"]))

    ie_data = result.get("interactiveElements")
    if ie_data and "questionId" in ie_data:
        ie_data["question_id"] = ie_data.pop("questionId")
    interactive = InteractiveElement(**ie_data) if ie_data else None

    # apply extracted answers to get updated QA for quality scoring
    updated_qa = _apply_extracted_answers(inp.current_qa, extracted)
    quality    = compute_quality_score(updated_qa)
    next_q     = _next_unanswered(updated_qa)

    logger.info("[EXTRACTOR] Extracted %d answers: %s", len(extracted),
                [(e.question_id, e.answer) for e in extracted])
    logger.info("[EXTRACTOR] Quality: %.1f%% (%s)", quality.overall, quality.grade)
    logger.info("[EXTRACTOR] Next unanswered: %s",
                next_q.question if next_q else "NONE — all done")
    logger.info("="*60)

    return state.model_copy(update={
        "extracted_answers":    extracted,
        "agent_message":        result.get("agentMessage", ""),
        "interactive_elements": interactive,
        "next_question":        next_q,
        "quality_score":        quality,
        "all_questions_answered": next_q is None,
    })


# ─────────────────────────────────────────────────────────────────────────────
# NODE 3 — EDUCATOR
# ─────────────────────────────────────────────────────────────────────────────

def education_node(state: GraphState) -> GraphState:
    """Answer a clarifying question then steer back to the profile."""
    logger.info("="*60)
    logger.info("[NODE: EDUCATOR] Answering clarifying question")
    inp      = state.agent_input
    qa_json  = json.dumps([_qa_to_dict(q) for q in inp.current_qa], indent=2)
    history  = _format_history(inp.conversation_history)
    next_q   = _next_unanswered(inp.current_qa)
    next_str = next_q.question if next_q else "No remaining questions."

    user_prompt = EDUCATION_USER.format(
        qa_json=qa_json,
        next_question=next_str,
        history=history,
        user_message=inp.user_message,
    )

    result  = _call_llm(EDUCATION_SYSTEM, user_prompt, temperature=0.5)
    quality = compute_quality_score(inp.current_qa)

    logger.info("[EDUCATOR] Education topic handled, returning to profile")
    logger.info("[EDUCATOR] Next question: %s", next_str)
    logger.info("="*60)

    return state.model_copy(update={
        "extracted_answers": [],
        "agent_message":     result.get("agentMessage", ""),
        "next_question":     next_q,
        "quality_score":     quality,
    })


# ─────────────────────────────────────────────────────────────────────────────
# NODE 4 — COMPLETION
# ─────────────────────────────────────────────────────────────────────────────

def completion_node(state: GraphState) -> GraphState:
    """All questions answered — produce a completion/wrap-up message."""
    logger.info("="*60)
    logger.info("[NODE: COMPLETION] All questions answered, wrapping up")
    quality = compute_quality_score(state.agent_input.current_qa)
    logger.info("[COMPLETION] Final quality: %.1f%% (%s)", quality.overall, quality.grade)

    user_prompt = COMPLETION_USER.format(
        quality_score=quality.overall,
        grade=quality.grade,
        suggestions="; ".join(quality.suggestions),
    )

    result = _call_llm(COMPLETION_SYSTEM, user_prompt, temperature=0.4)

    logger.info("[COMPLETION] Done.")
    logger.info("="*60)

    return state.model_copy(update={
        "extracted_answers":      [],
        "agent_message":          result.get("agentMessage", ""),
        "all_questions_answered": True,
        "quality_score":          quality,
    })


# ─────────────────────────────────────────────────────────────────────────────
# NODE 5 — FILE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def file_extraction_node(state: GraphState) -> GraphState:
    """Handle file upload extraction results — summarise and map to Q&A."""
    logger.info("="*60)
    logger.info("[NODE: FILE EXTRACTOR] Processing file extraction results")
    inp      = state.agent_input
    qa_json  = json.dumps([_qa_to_dict(q) for q in inp.current_qa], indent=2)
    file_res = json.dumps(inp.file_extraction_results or [], indent=2)

    user_prompt = FILE_EXTRACTION_USER.format(
        qa_json=qa_json,
        file_extraction_results=file_res,
        user_message=inp.user_message,
    )

    result = _call_llm(FILE_EXTRACTION_SYSTEM, user_prompt)

    extracted = [
        ExtractedAnswer(question_id=e["questionId"], answer=e["answer"])
        for e in result.get("extractedAnswers", [])
        if e.get("answer") is not None
    ]

    updated_qa = _apply_extracted_answers(inp.current_qa, extracted)
    quality    = compute_quality_score(updated_qa)
    next_q     = _next_unanswered(updated_qa)

    logger.info("[FILE EXTRACTOR] Extracted %d answers from files", len(extracted))
    logger.info("[FILE EXTRACTOR] Quality: %.1f%%", quality.overall)
    logger.info("="*60)

    return state.model_copy(update={
        "extracted_answers":      extracted,
        "agent_message":          result.get("agentMessage", ""),
        "next_question":          next_q,
        "quality_score":          quality,
        "all_questions_answered": next_q is None,
        "file_extraction_applied": True,
    })


# ─────────────────────────────────────────────────────────────────────────────
# NODE 6 — DOCUMENT UPLOAD
# Reads raw document content and maps it to Q&A using user's prompt
# ─────────────────────────────────────────────────────────────────────────────

def document_upload_node(state: GraphState) -> GraphState:
    """Parse uploaded document content and map to Q&A based on user prompt."""
    logger.info("="*60)
    logger.info("[NODE: DOCUMENT UPLOAD] Mapping document content to Q&A")
    inp      = state.agent_input
    qa_json  = json.dumps([_qa_to_dict(q) for q in inp.current_qa], indent=2)

    upload_prompt = inp.upload_prompt or inp.user_message or "Extract all relevant information."

    user_prompt = DOCUMENT_UPLOAD_USER.format(
        product_profile_id=inp.product_profile_id,
        qa_json=qa_json,
        upload_prompt=upload_prompt,
        document_content=inp.uploaded_document_content or "",
    )

    result = _call_llm(DOCUMENT_UPLOAD_SYSTEM, user_prompt, temperature=0.2)

    extracted = [
        ExtractedAnswer(question_id=e["questionId"], answer=e["answer"])
        for e in result.get("extractedAnswers", [])
        if e.get("answer") is not None
    ]

    updated_qa = _apply_extracted_answers(inp.current_qa, extracted)
    quality    = compute_quality_score(updated_qa)
    next_q     = _next_unanswered(updated_qa)

    logger.info("[DOCUMENT UPLOAD] Extracted %d answers from document", len(extracted))
    logger.info("[DOCUMENT UPLOAD] Quality: %.1f%% (%s)", quality.overall, quality.grade)
    logger.info("[DOCUMENT UPLOAD] Next unanswered: %s",
                next_q.question if next_q else "NONE — all done")
    logger.info("="*60)

    return state.model_copy(update={
        "extracted_answers":      extracted,
        "agent_message":          result.get("agentMessage", ""),
        "next_question":          next_q,
        "quality_score":          quality,
        "all_questions_answered": next_q is None,
    })


# ─────────────────────────────────────────────────────────────────────────────
# NODE 7 — RESPONSE ASSEMBLER
# Final node — packages everything into AgentOutput
# ─────────────────────────────────────────────────────────────────────────────

def assembler_node(state: GraphState) -> GraphState:
    """No LLM call — just ensures state is clean for output extraction."""
    logger.info("[NODE: ASSEMBLER] Packaging final output")
    logger.info("[ASSEMBLER] Message length: %d chars", len(state.agent_message))
    logger.info("[ASSEMBLER] Extracted answers: %d", len(state.extracted_answers))
    logger.info("[ASSEMBLER] All answered: %s", state.all_questions_answered)
    return state
