"""Text-only LLM client for the openlux gateway (urllib only). Key from TEST_API_KEY."""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Mapping, Optional

URL = "https://api.openlux.ai/v1/chat/completions"
DEFAULT_MODEL = "gpt-5.4"


def parse_json(text):
    if isinstance(text, list):
        text = " ".join(i.get("text", str(i)) if isinstance(i, dict) else str(i) for i in text)
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(text).strip(), flags=re.I | re.S)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object in response")
    return json.loads(text[start : end + 1])


def chat(prompt: str, model: str = DEFAULT_MODEL, system: str | None = None,
         json_mode: bool = False, max_tokens: int = 1200, temperature: float = 0.0,
         retries=(0, 3, 10, 25, 50), timeout: int = 60) -> str:
    key = os.environ.get("TEST_API_KEY")
    if not key:
        raise RuntimeError("TEST_API_KEY missing")
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    body = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "reasoning_effort": "none",
        "messages": messages,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    last = None
    for attempt, delay in enumerate(retries):
        if delay:
            time.sleep(delay)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="replace"))
            content = payload["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = " ".join(i.get("text", "") for i in content if isinstance(i, dict))
            return content
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError,
                KeyError, json.JSONDecodeError, OSError) as exc:
            last = f"{type(exc).__name__}: {exc}"
    raise RuntimeError(last)


class LLMConfigurationError(RuntimeError):
    """Raised when optional LLM enrichment was requested without a key."""


class LLMRequestError(RuntimeError):
    """Raised after all retry attempts for an LLM request fail."""


class OpenAICompatibleClient:
    """Dependency-free JSON client for OpenAI-compatible chat endpoints."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 30.0,
        retries: int = 2,
        api_key_env: str = "OPENAI_API_KEY",
        opener: Optional[Any] = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get(api_key_env)
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.timeout = float(timeout)
        self.retries = max(0, int(retries))
        self._opener = opener or urllib.request.urlopen

    def _require_key(self) -> str:
        if not self.api_key:
            raise LLMConfigurationError("LLM enrichment requested but the configured API key is missing")
        return self.api_key

    def complete_json(self, system: str, user: str, *, model: Optional[str] = None, temperature: float = 0.0) -> Any:
        key = self._require_key()
        payload = {
            "model": model or self.model,
            "temperature": temperature,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + key,
                     "User-Agent": "lsda-ra-kb/1.0"},
            method="POST",
        )
        last: Optional[BaseException] = None
        for attempt in range(self.retries + 1):
            try:
                with self._opener(request, timeout=self.timeout) as response:
                    raw = response.read().decode("utf-8", errors="replace")
                data = json.loads(raw)
                content = data["choices"][0]["message"]["content"]
                if isinstance(content, list):
                    content = "".join(part.get("text", "") if isinstance(part, Mapping) else str(part) for part in content)
                return parse_json(str(content))
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                    KeyError, IndexError, ValueError, json.JSONDecodeError) as exc:
                last = exc
                if attempt < self.retries:
                    time.sleep(min(2.0**attempt, 8.0))
        raise LLMRequestError("LLM JSON request failed after retries") from last


def from_environment(**kwargs: Any) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(**kwargs)
