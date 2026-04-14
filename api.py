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
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

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
