"""
api.py — FastAPI backend for the Q&A Chat Agent
Run with: uvicorn api:app --reload --port 8000
"""

import io
import os
import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
# LLM Gateway key is used as the api_key for ChatOpenAI — set it so langchain doesn't complain
os.environ["OPENAI_API_KEY"] = os.getenv("LLM_GATEWAY_KEY", "")

from models import (
    AgentInput, AgentOutput, QAPair, FieldType, Category,
    ExtractedAnswer, InteractiveElement, QualityScore,
)
from graph import run_agent, stream_agent

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
for noisy in ("httpx", "httpcore", "openai", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SPP Agent API",
    description="Q&A Chat Agent backend for Smart Product Profile",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# SAMPLE Q&A
# ─────────────────────────────────────────────────────────────────────────────

def make_qa_list() -> list[QAPair]:
    return [
        # ── System Information ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    product_profile_id: str = "profile-001"
    current_qa: list[dict]
    user_message: str
    conversation_history: list[dict] = []
    uploaded_document_content: Optional[str] = None
    upload_prompt: Optional[str] = None


class ChatResponse(BaseModel):
    agent_message: str
    extracted_answers: list[dict] = []
    interactive_elements: Optional[dict] = None
    all_questions_answered: bool = False
    quality_score: Optional[dict] = None


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _parse_uploaded_file(file: UploadFile) -> str:
    """Read text from an uploaded file (pdf, docx, or text)."""
    suffix = Path(file.filename).suffix.lower()
    raw = file.file.read()

    if suffix == ".pdf":
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(raw))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    elif suffix == ".docx":
        import docx
        doc = docx.Document(io.BytesIO(raw))
        return "\n".join(para.text for para in doc.paragraphs).strip()
    else:
        return raw.decode("utf-8").strip()


def _build_agent_input(req: ChatRequest) -> AgentInput:
    """Convert ChatRequest into AgentInput with proper QAPair objects."""
    qa_pairs = [QAPair(**q) for q in req.current_qa]
    return AgentInput(
        product_profile_id=req.product_profile_id,
        current_qa=qa_pairs,
        user_message=req.user_message,
        conversation_history=req.conversation_history,
        uploaded_document_content=req.uploaded_document_content,
        upload_prompt=req.upload_prompt,
    )


def _output_to_response(output: AgentOutput) -> ChatResponse:
    """Convert AgentOutput to serializable ChatResponse."""
    extracted = [
        {"question_id": ea.question_id, "answer": ea.answer}
        for ea in output.extracted_answers
    ]
    ie = None
    if output.interactive_elements:
        ie = {
            "type": output.interactive_elements.type,
            "question_id": output.interactive_elements.question_id,
            "options": output.interactive_elements.options,
        }
    qs = None
    if output.quality_score:
        qs = output.quality_score.model_dump()

    return ChatResponse(
        agent_message=output.agent_message,
        extracted_answers=extracted,
        interactive_elements=ie,
        all_questions_answered=output.all_questions_answered,
        quality_score=qs,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/qa-template")
def get_qa_template():
    """Return the default Q&A template."""
    return [q.model_dump() for q in make_qa_list()]


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Non-streaming chat endpoint."""
    agent_input = _build_agent_input(req)
    output = run_agent(agent_input)
    return _output_to_response(output)


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """
    Streaming chat endpoint — returns server-sent events (SSE).
    Each event is a JSON line with node progress or the final result.
    """
    agent_input = _build_agent_input(req)

    def event_generator():
        final_state = {}
        for event in stream_agent(agent_input):
            node_name = event["node"]
            node_state = event["state"]

            # Collect final state
            for key, val in node_state.items():
                if val is not None and val != "" and val != []:
                    final_state[key] = val

            # Send node progress event
            progress = {"type": "node", "node": node_name}
            yield f"data: {json.dumps(progress)}\n\n"

        # Build final response from accumulated state
        extracted = []
        if "extracted_answers" in final_state:
            for ea in final_state["extracted_answers"]:
                if hasattr(ea, "question_id"):
                    extracted.append({"question_id": ea.question_id, "answer": ea.answer})
                elif isinstance(ea, dict):
                    extracted.append(ea)

        ie = None
        if "interactive_elements" in final_state and final_state["interactive_elements"]:
            el = final_state["interactive_elements"]
            if hasattr(el, "type"):
                ie = {"type": el.type, "question_id": el.question_id, "options": el.options}
            elif isinstance(el, dict):
                ie = el

        qs = None
        if "quality_score" in final_state and final_state["quality_score"]:
            q = final_state["quality_score"]
            qs = q.model_dump() if hasattr(q, "model_dump") else q

        result = {
            "type": "result",
            "agent_message": final_state.get("agent_message", ""),
            "extracted_answers": extracted,
            "interactive_elements": ie,
            "all_questions_answered": final_state.get("all_questions_answered", False),
            "quality_score": qs,
        }
        yield f"data: {json.dumps(result)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    prompt: str = Form("Extract all relevant information and map to profile questions."),
    product_profile_id: str = Form("profile-001"),
    current_qa: str = Form("[]"),
    conversation_history: str = Form("[]"),
):
    """Upload a document and have the agent extract answers from it."""
    try:
        doc_content = _parse_uploaded_file(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {e}")

    qa_list = json.loads(current_qa)
    history = json.loads(conversation_history)

    req = ChatRequest(
        product_profile_id=product_profile_id,
        current_qa=qa_list if qa_list else [q.model_dump() for q in make_qa_list()],
        user_message=prompt,
        conversation_history=history,
        uploaded_document_content=doc_content,
        upload_prompt=prompt,
    )

    agent_input = _build_agent_input(req)
    output = run_agent(agent_input)
    return _output_to_response(output)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
