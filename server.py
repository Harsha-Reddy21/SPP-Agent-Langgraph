"""
server.py — Terminal CLI for the Q&A Chat Agent
Run with: python server.py
(Requires OPENAI_API_KEY in environment)
"""

import logging
import os
import sys
from copy import deepcopy
from pathlib import Path

from dotenv import load_dotenv

from models import AgentInput, AgentOutput, QAPair, FieldType, Category
from graph import run_agent, build_graph

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING — DEBUG level to see all internal node/LLM activity
# ─────────────────────────────────────────────────────────────────────────────

os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")  # Ensure key is in env for any subprocesses
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
        # ── System Information ─────────────────────────────────────────────
        QAPair(
            id="qa-1", question="Select the appropriate option.",
            answer=None, field_type=FieldType.RADIO,
            options=[
                "I am a submitter or custodian of a system that has completed AI Reviews, and my system has had a scope change that requires rereview",
                "I would like to create a new submission",
            ],
            category=Category.SOLUTION, required=True, sort_order=1,
            help_text="Choose whether this is a new submission or a rereview of an existing system.",
        ),
        QAPair(
            id="qa-2", question="What is the title of the System?",
            answer=None, field_type=FieldType.TEXT,
            category=Category.SOLUTION, required=True, sort_order=2,
            help_text="Provide the official name of the system or solution.",
        ),
        QAPair(
            id="qa-3", question="Provide an overview of the solution.",
            answer=None, field_type=FieldType.TEXTAREA,
            category=Category.SOLUTION, required=True, sort_order=3,
            help_text="Describe the solution, its purpose, and how it works at a high level.",
        ),
        QAPair(
            id="qa-4", question="What is the value proposition for this solution?",
            answer=None, field_type=FieldType.TEXTAREA,
            category=Category.SOLUTION, required=True, sort_order=4,
            help_text="Explain the business value and benefits this solution provides.",
        ),
        QAPair(
            id="qa-5", question="Which organization will sponsor the System?",
            answer=None, field_type=FieldType.DROPDOWN,
            options=[
                "Lilly Research Laboratories", "Lilly Oncology", "Lilly Neuroscience",
                "Lilly Immunology", "Lilly Diabetes", "Manufacturing Operations",
                "Global Quality", "IT / Tech@Lilly", "Finance", "Human Resources",
                "Legal", "Marketing", "Supply Chain", "Commercial", "Medical Affairs",
                "Regulatory", "Corporate Affairs", "Other",
            ],
            category=Category.SOLUTION, required=True, sort_order=5,
            help_text="Select the organization that will own and sponsor this system.",
        ),
        QAPair(
            id="qa-6", question="Are you planning to implement a vendor product as part or all of this system?",
            answer=None, field_type=FieldType.RADIO,
            options=["Yes", "No"],
            category=Category.SOLUTION, required=True, sort_order=6,
            help_text="Indicate if an external vendor product is involved.",
        ),
        QAPair(
            id="qa-7", question="Which vendor(s)?",
            answer=None, field_type=FieldType.DROPDOWN,
            options=[
                "Microsoft", "Google", "Amazon Web Services", "IBM", "Salesforce",
                "SAP", "Oracle", "Palantir", "Databricks", "Snowflake",
                "Accenture", "Deloitte", "Cognizant", "Infosys", "Other",
            ],
            category=Category.SOLUTION, required=False, sort_order=7,
            help_text="Select the vendor(s) involved. Only applicable if vendor product is being used.",
        ),
        QAPair(
            id="qa-8", question="What product(s)?",
            answer=None, field_type=FieldType.TEXT,
            category=Category.SOLUTION, required=False, sort_order=8,
            help_text="Name of the vendor product(s) being implemented.",
        ),

        # ── User Information ───────────────────────────────────────────────
        QAPair(
            id="qa-9", question="Briefly describe the users of the system, what they will be using it for, and what problem it will solve for them.",
            answer=None, field_type=FieldType.TEXTAREA,
            category=Category.USER, required=True, sort_order=1,
            help_text="Describe who will use the system, their use case, and the problem being solved.",
        ),
        QAPair(
            id="qa-10", question="(Select all that apply) Where will you deploy your System?",
            answer=None, field_type=FieldType.MULTI_SELECT,
            options=[
                "United States", "Europe", "Asia Pacific", "Japan", "China",
                "Latin America", "Canada", "Middle East & Africa", "India", "Global",
            ],
            category=Category.USER, required=True, sort_order=2,
            help_text="Select all regions where the system will be deployed.",
        ),
        QAPair(
            id="qa-11", question="(Select all that apply) Who is the audience of this system?",
            answer=None, field_type=FieldType.MULTI_SELECT,
            options=[
                "Internal Employees", "External Customers", "Healthcare Professionals",
                "Patients", "Partners / Vendors", "Regulators", "Clinical Trial Sites",
                "Researchers", "General Public", "Other",
            ],
            category=Category.USER, required=True, sort_order=3,
            help_text="Select all audiences who will interact with the system.",
        ),

        # ── Technical Information ──────────────────────────────────────────
        QAPair(
            id="qa-12", question="Do you have any additional technical information to provide?",
            answer=None, field_type=FieldType.RADIO,
            options=["Yes", "No"],
            category=Category.TECHNICAL, required=False, sort_order=1,
            help_text="If yes, please fill in the technical details below. These details will be confirmed during the AI Technical Review.",
        ),
        QAPair(
            id="qa-13", question="What is the maturity of the System?",
            answer=None, field_type=FieldType.DROPDOWN,
            options=[
                "Concept / Ideation", "Proof of Concept (POC)", "Pilot",
                "Minimum Viable Product (MVP)", "Production", "Scaling / Expansion",
                "Maintenance / Steady State", "Decommissioning",
            ],
            category=Category.TECHNICAL, required=False, sort_order=2,
            help_text="Select the current maturity stage of the system.",
        ),
        QAPair(
            id="qa-14", question="(Select all that apply) Does the System process any of the following data?",
            answer=None, field_type=FieldType.MULTI_SELECT,
            options=[
                "Personal Information (PI)", "Sensitive Personal Information (SPI)",
                "Protected Health Information (PHI)", "Patient Data",
                "Clinical Trial Data", "Genomic / Genetic Data",
                "Financial Data", "Employee Data", "Intellectual Property",
                "Commercial / Sales Data", "Manufacturing Data", "None of the above",
            ],
            category=Category.TECHNICAL, required=False, sort_order=3,
            help_text="Processing includes collecting, recording, organizing, storing, using, transferring, viewing, combining, or deleting data.",
        ),
        QAPair(
            id="qa-15", question="What is the highest data classification for processed data?",
            answer=None, field_type=FieldType.DROPDOWN,
            options=[
                "Public", "Internal Use", "Confidential", "Restricted",
            ],
            category=Category.TECHNICAL, required=False, sort_order=4,
            help_text="If data classification increases during implementation, a new review cycle will be required.",
        ),
        QAPair(
            id="qa-16", question="Is any of this data used for training the model?",
            answer=None, field_type=FieldType.RADIO,
            options=["Yes", "No", "Not Sure"],
            category=Category.TECHNICAL, required=False, sort_order=5,
            help_text="Indicate whether the processed data is used to train or fine-tune AI models.",
        ),
        QAPair(
            id="qa-17", question="Describe the Data Needed.",
            answer=None, field_type=FieldType.TEXTAREA,
            category=Category.TECHNICAL, required=False, sort_order=6,
            help_text="Describe what data the system needs, its sources, and how it will be used.",
        ),
        QAPair(
            id="qa-18", question="(Select all that apply) Does the AI Functionality of the system do any of the following?",
            answer=None, field_type=FieldType.MULTI_SELECT,
            options=[
                "Content Generation (text, images, code)",
                "Summarization", "Classification / Categorization",
                "Prediction / Forecasting", "Recommendation",
                "Natural Language Processing (NLP)", "Computer Vision",
                "Speech Recognition / Synthesis", "Anomaly Detection",
                "Data Extraction / Parsing", "Decision Support",
                "Process Automation (RPA + AI)", "Search / Retrieval (RAG)",
                "Translation", "Sentiment Analysis", "Other",
            ],
            category=Category.TECHNICAL, required=False, sort_order=7,
            help_text="Select all AI capabilities that the system uses.",
        ),
        QAPair(
            id="qa-19", question="Is the AI System output reviewed by a human before it is used?",
            answer=None, field_type=FieldType.RADIO,
            options=[
                "Yes \u2013 100% of the output",
                "Yes \u2013 a sampling (please describe)",
                "No",
            ],
            category=Category.TECHNICAL, required=False, sort_order=8,
            help_text="Indicate the level of human review applied to AI output.",
        ),
        QAPair(
            id="qa-20", question="Which AI platforms/technologies will be used?",
            answer=None, field_type=FieldType.MULTI_SELECT,
            options=[
                "Azure OpenAI Service", "Amazon Bedrock", "Amazon SageMaker",
                "Google Vertex AI", "Google Cloud AI Platform", "IBM Watson",
                "Databricks ML", "Snowflake Cortex", "Palantir Foundry",
                "Hugging Face", "Custom / In-house Platform", "Other",
            ],
            category=Category.TECHNICAL, required=False, sort_order=9,
            help_text="Select the AI platform(s) or technology stack being used.",
        ),
        QAPair(
            id="qa-21", question="Which Models will be used?",
            answer=None, field_type=FieldType.MULTI_SELECT,
            options=[
                "GPT-4o", "GPT-4", "GPT-3.5 Turbo", "Claude 3.5 Sonnet",
                "Claude 3 Opus", "Gemini 1.5 Pro", "Gemini 1.5 Flash",
                "Cohere Command R+", "Custom Fine-tuned Model", "Other",
            ],
            category=Category.TECHNICAL, required=False, sort_order=10,
            help_text="Select the AI model(s) that will be used.",
        ),
        QAPair(
            id="qa-22", question="Which Open Source Models will be used?",
            answer=None, field_type=FieldType.MULTI_SELECT,
            options=[
                "Llama 3", "Llama 2", "Mistral", "Mixtral",
                "Falcon", "BLOOM", "Stable Diffusion",
                "Whisper", "BERT", "RoBERTa", "T5", "None", "Other",
            ],
            category=Category.TECHNICAL, required=False, sort_order=11,
            help_text="Select any open source models involved.",
        ),
        QAPair(
            id="qa-23", question="Does the AI/ML use continuous learning?",
            answer=None, field_type=FieldType.RADIO,
            options=["Yes", "No"],
            category=Category.TECHNICAL, required=False, sort_order=12,
            help_text="Indicate if the model continuously learns and updates from new data.",
        ),

        # ── General / Administrative ───────────────────────────────────────
        QAPair(
            id="qa-24", question="Privacy Request Identifier",
            answer=None, field_type=FieldType.TEXT,
            category=Category.GENERAL, required=False, sort_order=1,
            help_text="Enter the Privacy Request Identifier if a privacy review has been initiated.",
        ),
        QAPair(
            id="qa-25", question="SAE review number",
            answer=None, field_type=FieldType.TEXT,
            category=Category.GENERAL, required=False, sort_order=2,
            help_text="Enter the SAE review number if applicable.",
        ),
        QAPair(
            id="qa-26", question="WwTP review",
            answer=None, field_type=FieldType.TEXT,
            category=Category.GENERAL, required=False, sort_order=3,
            help_text="Enter the WwTP review reference if applicable.",
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


def read_document(file_path: str) -> str:
    """Read text content from a file. Supports .txt, .pdf, .docx."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            raise ImportError("Install PyPDF2 to read PDF files: pip install PyPDF2")
        reader = PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text.strip()

    elif suffix == ".docx":
        try:
            import docx
        except ImportError:
            raise ImportError("Install python-docx to read DOCX files: pip install python-docx")
        doc = docx.Document(str(path))
        text = "\n".join(para.text for para in doc.paragraphs)
        return text.strip()

    else:
        # treat as plain text (.txt, .md, .csv, etc.)
        return path.read_text(encoding="utf-8").strip()


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
    print("#  Type 'upload <filepath>' to upload a document.")
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

        # ── Document upload handling ──────────────────────────────────
        uploaded_doc_content = None
        upload_prompt = None
        if user_message.lower().startswith("upload "):
            file_path = user_message[7:].strip().strip('"').strip("'")
            try:
                uploaded_doc_content = read_document(file_path)
                print(f"\n  Document loaded: {file_path} ({len(uploaded_doc_content)} chars)")
            except (FileNotFoundError, ImportError) as e:
                print(f"\n  [ERROR] {e}")
                continue

            try:
                upload_prompt = input("  Your instructions for this document: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not upload_prompt:
                upload_prompt = "Extract all relevant information from this document and map it to the product profile questions."

            user_message = upload_prompt

        logger.info("*" * 60)
        logger.info("[TURN] User said: %s", user_message)
        logger.info("*" * 60)

        try:
            agent_input = AgentInput(
                product_profile_id=profile_id,
                current_qa=deepcopy(qa),
                user_message=user_message,
                conversation_history=history,
                uploaded_document_content=uploaded_doc_content,
                upload_prompt=upload_prompt,
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
