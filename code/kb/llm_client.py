"""Text-only LLM client for the openlux gateway (urllib only). Key from TEST_API_KEY."""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

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
