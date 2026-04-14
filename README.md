# Q&A Chat Agent — LangGraph Implementation

Conversational form-filling agent that chats with users to fill a structured
product profile, extracts answers in real time, and scores profile quality.

---

## Architecture

```
User Message
     │
     ▼
┌─────────┐     route="extract"      ┌─────────────┐
│ Router  │ ──────────────────────► │  Extractor  │
│  Node   │     route="educate"      ├─────────────┤
│         │ ──────────────────────► │  Educator   │
│         │     route="complete"     ├─────────────┤
│         │ ──────────────────────► │  Completion │
│         │   route="file_extract"   ├─────────────┤
└─────────┘ ──────────────────────► │File Extract │
                                     └──────┬──────┘
                                            │
                                     ┌──────▼──────┐
                                     │  Assembler  │
                                     └──────┬──────┘
                                            │
                                     ┌──────▼──────┐
                                     │  AgentOutput│
                                     └─────────────┘
```

### Nodes

| Node | Purpose |
|------|---------|
| **Router** | Classifies user intent → extract / educate / complete / file_extraction |
| **Extractor** | Pulls answers from free text, picks next question, includes interactive elements |
| **Educator** | Answers clarifying questions, then steers back to the form |
| **Completion** | Wraps up when all Q&A pairs are answered |
| **File Extractor** | Summarises file-upload extraction results and maps to Q&A |
| **Assembler** | Packages final state into `AgentOutput` |

### Quality Scoring

Scores are computed after every extraction pass:

- **Required fields** → 70% weight per category
- **Optional fields** → 30% weight per category
- **Category weights**: Solution 40% · User 25% · Technical 25% · General 10%
- **Richness bonus**: Free-text answers >20 words earn up to +5 pts per category
- **Grades**: Excellent (≥90) · Good (≥70) · Fair (≥50) · Poor (<50)

---

## File Structure

```
chat_agent/
├── models.py          # All Pydantic models + LangGraph State
├── prompts.py         # All LLM prompt templates
├── nodes.py           # One function per graph node
├── quality.py         # Quality scoring logic
├── graph.py           # Graph wiring + run_agent() public API
├── server.py          # FastAPI HTTP server
├── test_agent.py      # End-to-end scenario tests
└── requirements.txt
```

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set API key
export ANTHROPIC_API_KEY=sk-ant-...

# 3a. Run the HTTP server
python server.py

# 3b. OR run the scenario tests directly
python test_agent.py
```

---

## API

### POST /chat

**Request:**
```json
{
  "product_profile_id": "uuid",
  "current_qa": [
    {
      "id": "qa-uuid-1",
      "question": "What is the product title?",
      "answer": null,
      "field_type": "text",
      "category": "solution",
      "required": true,
      "sort_order": 1
    }
  ],
  "user_message": "We plan to deploy in US East using GPT-4.",
  "conversation_history": [],
  "file_extraction_results": null
}
```

**Response:**
```json
{
  "extracted_answers": [
    {"questionId": "qa-uuid-2", "answer": "US East"},
    {"questionId": "qa-uuid-3", "answer": "GPT-4"}
  ],
  "agent_message": "Got it! Deploying in US East with GPT-4. Next — who are the primary users?",
  "interactive_elements": null,
  "all_questions_answered": false,
  "quality_score": {
    "overall": 42.5,
    "required_completion": 40.0,
    "optional_completion": 0.0,
    "category_scores": {"solution": 0.0, "user": 0.0, "technical": 100.0, "general": 0.0},
    "grade": "Poor",
    "suggestions": ["Complete more Solution section fields to improve your score."]
  }
}
```

---

## Client-Side Responsibilities

The agent is **stateless** — the client must:

1. **Store the full `current_qa` list** and send it every turn.
2. **Apply `extracted_answers`** to its local Q&A store after each response.
3. **Maintain `conversation_history`** as `[{"role": "user"|"assistant", "content": "..."}]`
   and append both sides each turn before the next call.
4. **Render `interactive_elements`** as dropdown/radio cards in the chat UI.
5. **Show `quality_score`** in the Preview panel and update it after each turn.
6. **Set `qa.skipped = true`** when the user clicks "Ignore for now" and include
   that in the next `current_qa` payload.

---

## Covered Scenarios

| # | Scenario |
|---|----------|
| 1 | Agent receives Q&A pairs as structured input |
| 2 | Agent returns structured JSON output |
| 3 | Smart next-question selection (required → category → sort) |
| 4 | Bulk answer extraction from free-text messages |
| 5 | Dropdown/radio constrained field handling |
| 6 | "Ignore for now" skip handling |
| 7 | No-information messages handled gracefully |
| 8 | Completion detection with `allQuestionsAnswered` flag |
| 9 | Answer correction / overwrite |
| 10 | File extraction result summarisation |
