"""
server.py — Terminal CLI for the Q&A Chat Agent
Run with: python server.py
(Requires OPENAI_API_KEY in environment)
"""

import logging
import os
import sys
from copy import deepcopy

from dotenv import load_dotenv

from models import AgentInput, AgentOutput, QAPair, FieldType, Category
from graph import run_agent, build_graph

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING — DEBUG level to see all internal node/LLM activity
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Quiet down noisy third-party loggers
for noisy in ("httpx", "httpcore", "openai", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# SAMPLE Q&A — same fixture used in tests
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


def apply_answers(qa_list: list[QAPair], extracted: list) -> list[QAPair]:
    """Merge extracted answers into QA list."""
    qa_list = deepcopy(qa_list)
    answer_map = {e.question_id: e.answer for e in extracted}
    for qa in qa_list:
        if qa.id in answer_map:
            qa.answer = answer_map[qa.id]
    return qa_list


def print_qa_status(qa_list: list[QAPair]):
    """Print current Q&A state."""
    print("\n" + "-" * 60)
    print("  CURRENT Q&A STATUS")
    print("-" * 60)
    for q in qa_list:
        status = "ANSWERED" if q.answer is not None else ("SKIPPED" if q.skipped else "PENDING")
        answer_display = f" -> {q.answer}" if q.answer is not None else ""
        print(f"  [{status:>8}] {q.id}: {q.question}{answer_display}")
    print("-" * 60)


def print_output(output: AgentOutput):
    """Pretty-print the agent response."""
    print("\n" + "=" * 60)
    print("  AGENT RESPONSE")
    print("=" * 60)
    print(f"\n  {output.agent_message}\n")
    if output.extracted_answers:
        print("  Extracted answers:")
        for ea in output.extracted_answers:
            print(f"    [{ea.question_id}] = {ea.answer}")
    if output.interactive_elements:
        ie = output.interactive_elements
        print(f"  Interactive ({ie.type}): {ie.options}")
    if output.quality_score:
        qs = output.quality_score
        print(f"\n  Quality: {qs.overall}% ({qs.grade}) | "
              f"Required: {qs.required_completion}% | Optional: {qs.optional_completion}%")
        if qs.suggestions:
            print(f"  Suggestions: {'; '.join(qs.suggestions)}")
    if output.all_questions_answered:
        print("\n  ALL QUESTIONS ANSWERED")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN INTERACTIVE LOOP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set. Export it or add to .env file.")
        sys.exit(1)

    print("\n" + "#" * 60)
    print("#  Q&A Chat Agent — Terminal Mode")
    print("#  Type your messages below. Type 'quit' or 'exit' to stop.")
    print("#  Type 'status' to see current Q&A state.")
    print("#" * 60)

    logger.info("Building LangGraph compiled graph...")
    build_graph()
    logger.info("Graph ready. Starting interactive session.\n")

    qa = make_qa_list()
    history: list[dict] = []
    profile_id = "profile-cli-001"

    print_qa_status(qa)

    while True:
        try:
            user_message = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_message:
            continue
        if user_message.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if user_message.lower() == "status":
            print_qa_status(qa)
            continue

        logger.info("*" * 60)
        logger.info("[TURN] User said: %s", user_message)
        logger.info("*" * 60)

        try:
            agent_input = AgentInput(
                product_profile_id=profile_id,
                current_qa=deepcopy(qa),
                user_message=user_message,
                conversation_history=history,
            )
            output: AgentOutput = run_agent(agent_input)
        except Exception as e:
            logger.exception("Agent error: %s", e)
            print(f"\n  [ERROR] {e}")
            continue

        # Apply extracted answers to local Q&A state
        if output.extracted_answers:
            qa = apply_answers(qa, output.extracted_answers)

        # Update conversation history
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": output.agent_message})

        # Display output
        print_output(output)

        if output.all_questions_answered:
            print("\nProfile complete! You can type 'status' to review or 'quit' to exit.")


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
