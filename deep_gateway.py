"""
Python equivalent of llmService.js
Requires: pip install requests
"""

import json
import logging
import os
import time
import threading
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config (mirrors your .env variables) ─────────────────────────────────────
TENANT_ID       = os.getenv("Tenant_ID")
CLIENT_ID       = os.getenv("Application_ID")
CLIENT_SECRET   = os.getenv("Client_Secret_value")
LLM_GATEWAY_KEY = os.getenv("LLM_GATEWAY_KEY")
BASE_URL        = os.getenv("LLM_BASE_URL")
MODEL           = os.getenv("LLM_MODEL")
SCOPE           = "api://llm-gateway.lilly.com/.default"


# ── Auth (equivalent to authService.js) ──────────────────────────────────────
class AuthService:
    def __init__(self):
        self._token: str | None = None
        self._expiry: float = 0
        self._lock = threading.Lock()

    def get_access_token(self, force_refresh: bool = False) -> str:
        with self._lock:
            if force_refresh or not self._token or time.time() >= self._expiry:
                logger.info("🔑 Fetching new OAuth2 token...")
                r = requests.post(
                    f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token",
                    data={
                        "grant_type":    "client_credentials",
                        "client_id":     CLIENT_ID,
                        "client_secret": CLIENT_SECRET,
                        "scope":         SCOPE,
                    },
                    timeout=30,
                )
                r.raise_for_status()
                data = r.json()
                self._token = data["access_token"]
                self._expiry = time.time() + data.get("expires_in", 3600) - 60
                logger.info("✅ Token obtained.")
            return self._token


auth_service = AuthService()


# ── LLM Service (equivalent to LLMService class in llmService.js) ────────────
class LLMService:
    def query_llm(
        self,
        prompt: str = "What is the capital of France?",
        streaming: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ):
        logger.info(f"🤖 Querying LLM with prompt: {prompt[:50]}...")
        logger.info(f"📡 Streaming enabled: {streaming}")

        access_token = auth_service.get_access_token()
        url = BASE_URL.rstrip("/") + "/chat/completions"

        logger.info(f"🌐 Making request to: {url}")
        logger.info(f"📊 Model: {MODEL}")

        headers = {
            "Authorization":     f"Bearer {access_token}",
            "X-LLM-Gateway-Key": LLM_GATEWAY_KEY,
            "Content-Type":      "application/json",
        }
        payload = {
            "model":       MODEL,
            "messages":    [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens":  max_tokens,
            "stream":      streaming,
        }

        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=30,
                stream=streaming,
                verify=False,   # mirrors rejectUnauthorized: false in JS
            )
            response.raise_for_status()
            logger.info("✅ LLM Response received successfully")

            # ── Streaming ─────────────────────────────────────────────────
            if streaming:
                logger.info("🔄 Streaming response:")
                full_text = ""
                for line in response.iter_lines():
                    if not line:
                        continue
                    decoded = line.decode("utf-8")
                    if decoded.startswith("data: "):
                        decoded = decoded[6:]
                    if decoded.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(decoded)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        print(delta, end="", flush=True)
                        full_text += delta
                    except (json.JSONDecodeError, KeyError):
                        continue
                print()  # newline after stream
                return {"text": full_text, "model": MODEL, "streaming": True}

            # ── Non-streaming ─────────────────────────────────────────────
            data = response.json()
            logger.info(f"📋 Response:\n{json.dumps(data, indent=2)}")
            return {
                "choices":   data["choices"],
                "model":     data.get("model"),
                "usage":     data.get("usage"),
                "streaming": False,
            }

        except requests.HTTPError as e:
            logger.error(f"❌ HTTP error: {e.response.status_code} {e.response.text[:300]}")
            raise RuntimeError(f"LLM Query Failed: {e}")
        except Exception as e:
            logger.error(f"❌ Error querying LLM: {e}")
            raise RuntimeError(f"LLM Query Failed: {e}")


llm_service = LLMService()


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Non-streaming
    result = llm_service.query_llm("What is the mechanism of action of metformin?", streaming=False)
    print("\n=== Response ===")
    print(result["choices"][0]["message"]["content"])

    # Uncomment to test streaming:
    # result = llm_service.query_llm("Tell me about Eli Lilly", streaming=True)