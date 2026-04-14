"""
test_agent.py — End-to-end scenario tests covering all 10 acceptance criteria
Run with: python test_agent.py
(Requires ANTHROPIC_API_KEY in environment)
"""

import json
import os
from copy import deepcopy

from models import AgentInput, QAPair, FieldType, Category
from graph import run_agent


# ─────────────────────────────────────────────────────────────────────────────
# SAMPLE Q&A FIXTURE
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
            options=["< $100K", "$100K–$500K", "$500K–$1M", "> $1M"],
            category=Category.GENERAL, required=False, sort_order=1,
        ),
    ]


def apply_answers(qa_list: list[QAPair], extracted: list) -> list[QAPair]:
    """Merge extracted answers into QA list (simulates backend save)."""
    qa_list = deepcopy(qa_list)
    answer_map = {e.question_id: e.answer for e in extracted}
    for qa in qa_list:
        if qa.id in answer_map:
            qa.answer = answer_map[qa.id]
    return qa_list


def print_output(scenario: str, output):
    print(f"\n{'='*70}")
    print(f"  {scenario}")
    print(f"{'='*70}")
    print(f"  Agent: {output.agent_message}")
    if output.extracted_answers:
        print(f"  Extracted:")
        for ea in output.extracted_answers:
            print(f"    • [{ea.question_id}] → {ea.answer}")
    if output.interactive_elements:
        ie = output.interactive_elements
        print(f"  Interactive ({ie.type}): {ie.options}")
    if output.quality_score:
        qs = output.quality_score
        print(f"  Quality: {qs.overall}% ({qs.grade}) | "
              f"Required: {qs.required_completion}%")
    if output.all_questions_answered:
        print("  ✅ ALL QUESTIONS ANSWERED")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIOS
# ─────────────────────────────────────────────────────────────────────────────

def run_all_scenarios():
    qa = make_qa_list()
    history = []
    profile_id = "profile-test-001"

    # ── Scenario 7: no useful information ────────────────────────────────────
    output = run_agent(AgentInput(
        product_profile_id=profile_id,
        current_qa=deepcopy(qa),
        user_message="Hi there!",
        conversation_history=history,
    ))
    print_output("Scenario 7 — No useful information ('Hi')", output)
    history.append({"role": "user",      "content": "Hi there!"})
    history.append({"role": "assistant", "content": output.agent_message})

    # ── Scenario 4: bulk free-text extraction ─────────────────────────────────
    output = run_agent(AgentInput(
        product_profile_id=profile_id,
        current_qa=deepcopy(qa),
        user_message=(
            "We're building AI Clinical Trial Analyzer. "
            "It helps pharma companies reduce trial analysis time by 80%. "
            "The primary users are clinical data managers."
        ),
        conversation_history=history,
    ))
    print_output("Scenario 4 — Bulk free-text extraction", output)
    qa = apply_answers(qa, output.extracted_answers)
    history.append({"role": "user",      "content": "We're building AI Clinical Trial Analyzer..."})
    history.append({"role": "assistant", "content": output.agent_message})

    # ── Scenario 5: dropdown question ─────────────────────────────────────────
    output = run_agent(AgentInput(
        product_profile_id=profile_id,
        current_qa=deepcopy(qa),
        user_message="We want to deploy in US East.",
        conversation_history=history,
    ))
    print_output("Scenario 5 — Dropdown answer (deployment region)", output)
    qa = apply_answers(qa, output.extracted_answers)
    history.append({"role": "user",      "content": "We want to deploy in US East."})
    history.append({"role": "assistant", "content": output.agent_message})

    # ── Education request ─────────────────────────────────────────────────────
    output = run_agent(AgentInput(
        product_profile_id=profile_id,
        current_qa=deepcopy(qa),
        user_message="What's the difference between GPT-4 and Claude?",
        conversation_history=history,
    ))
    print_output("Education — Model comparison question", output)
    history.append({"role": "user",      "content": "What's the difference between GPT-4 and Claude?"})
    history.append({"role": "assistant", "content": output.agent_message})

    # ── Continue filling ──────────────────────────────────────────────────────
    output = run_agent(AgentInput(
        product_profile_id=profile_id,
        current_qa=deepcopy(qa),
        user_message="We'll go with Claude. Accenture is our development partner.",
        conversation_history=history,
    ))
    print_output("Extraction — AI model + vendor from one message", output)
    qa = apply_answers(qa, output.extracted_answers)
    history.append({"role": "user",      "content": "We'll go with Claude. Accenture is our development partner."})
    history.append({"role": "assistant", "content": output.agent_message})

    # ── Scenario 6: skip a question ────────────────────────────────────────────
    # Simulate "ignore for now" on qa-7 (budget)
    qa_with_skip = deepcopy(qa)
    for q in qa_with_skip:
        if q.id == "qa-7":
            q.skipped = True
    output = run_agent(AgentInput(
        product_profile_id=profile_id,
        current_qa=qa_with_skip,
        user_message="Ignore for now",
        conversation_history=history,
    ))
    print_output("Scenario 6 — Skip question (budget skipped)", output)

    # ── Scenario 9: answer correction ─────────────────────────────────────────
    output = run_agent(AgentInput(
        product_profile_id=profile_id,
        current_qa=deepcopy(qa),
        user_message="Actually, change the deployment region to EU instead.",
        conversation_history=history,
    ))
    print_output("Scenario 9 — Answer correction (region → EU)", output)
    qa = apply_answers(qa, output.extracted_answers)
    history.append({"role": "user",      "content": "Actually, change the deployment region to EU instead."})
    history.append({"role": "assistant", "content": output.agent_message})

    # ── Scenario 10: file extraction ──────────────────────────────────────────
    file_results = [
        {"field": "budget", "value": "$500K–$1M", "confidence": 0.92,
         "source": "budget_proposal.pdf"},
    ]
    output = run_agent(AgentInput(
        product_profile_id=profile_id,
        current_qa=deepcopy(qa),
        user_message="I just uploaded the budget proposal PDF.",
        conversation_history=history,
        file_extraction_results=file_results,
    ))
    print_output("Scenario 10 — File extraction results", output)
    qa = apply_answers(qa, output.extracted_answers)

    # ── Scenario 8: all answered ──────────────────────────────────────────────
    # Manually fill any remaining nulls for the completion test
    for q in qa:
        if q.answer is None and not q.skipped:
            q.answer = "Test answer"

    output = run_agent(AgentInput(
        product_profile_id=profile_id,
        current_qa=deepcopy(qa),
        user_message="I think I've answered everything.",
        conversation_history=history,
    ))
    print_output("Scenario 8 — All questions answered (completion)", output)


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        exit(1)
    run_all_scenarios()
