"""
models.py — Pydantic data models for the Q&A Chat Agent
"""

from typing import Any, Optional
from pydantic import BaseModel, Field
from enum import Enum


class FieldType(str, Enum):
    TEXT = "text"
    DROPDOWN = "dropdown"
    RADIO = "radio"
    MULTI_SELECT = "multi_select"
    NUMBER = "number"
    DATE = "date"
    TEXTAREA = "textarea"


class Category(str, Enum):
    SOLUTION = "solution"
    USER = "user"
    TECHNICAL = "technical"
    GENERAL = "general"


# ── Input Models ──────────────────────────────────────────────────────────────

class QAPair(BaseModel):
    id: str
    question: str
    answer: Optional[Any] = None
    field_type: FieldType = FieldType.TEXT
    options: Optional[list[str]] = None          # for dropdown/radio/multi_select
    category: Category = Category.GENERAL
    required: bool = True
    sort_order: int = 0
    help_text: Optional[str] = None              # educational context for the field
    skipped: bool = False                        # user said "ignore for now"


class AgentInput(BaseModel):
    product_profile_id: str
    current_qa: list[QAPair]
    user_message: str
    conversation_history: list[dict] = Field(default_factory=list)
    file_extraction_results: Optional[list[dict]] = None  # from file upload (SPP-009)


# ── Output Models ─────────────────────────────────────────────────────────────

class ExtractedAnswer(BaseModel):
    question_id: str
    answer: Any


class InteractiveElement(BaseModel):
    type: str                   # "dropdown" | "radio" | "multi_select"
    question_id: str
    options: list[str]


class AgentOutput(BaseModel):
    extracted_answers: list[ExtractedAnswer] = Field(default_factory=list)
    agent_message: str
    interactive_elements: Optional[InteractiveElement] = None
    all_questions_answered: bool = False
    quality_score: Optional["QualityScore"] = None


class QualityScore(BaseModel):
    overall: float                          # 0.0 – 100.0
    required_completion: float              # % of required fields answered
    optional_completion: float             # % of optional fields answered
    category_scores: dict[str, float]      # per-category breakdown
    grade: str                             # "Excellent" / "Good" / "Fair" / "Poor"
    suggestions: list[str]                 # what to improve


AgentOutput.model_rebuild()


# ── LangGraph State ───────────────────────────────────────────────────────────

class GraphState(BaseModel):
    # ── inputs
    agent_input: AgentInput

    # ── working state (mutated by nodes)
    extracted_answers: list[ExtractedAnswer] = Field(default_factory=list)
    next_question: Optional[QAPair] = None
    agent_message: str = ""
    interactive_elements: Optional[InteractiveElement] = None
    all_questions_answered: bool = False
    quality_score: Optional[QualityScore] = None
    is_education_request: bool = False
    education_topic: Optional[str] = None
    file_extraction_applied: bool = False

    # ── routing flags
    route: str = "extract"   # extract | educate | complete | file_extraction

    class Config:
        arbitrary_types_allowed = True
