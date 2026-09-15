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
         reasoning_effort: str | None = None, retries=(0, 3, 10, 25, 50), timeout: int = 60) -> str:
    """Text completion via openlux.

    reasoning_effort is only sent when explicitly provided (reasoning models reject it otherwise).
    json_mode injects a system rule containing the literal word 'json' (required by OpenAI).
    """
    key = os.environ.get("TEST_API_KEY")
    if not key:
        raise RuntimeError("TEST_API_KEY missing")
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    if json_mode:
        messages.append({"role": "system", "content": "You must output only a valid json object."})
    messages.append({"role": "user", "content": prompt})
    body = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if reasoning_effort is not None:
        body["reasoning_effort"] = reasoning_effort
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
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                last = f"HTTPError: {exc}"
                time.sleep(5 * (attempt + 1))
                continue
            if 400 <= exc.code < 500:
                raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:300]}")
            last = f"HTTPError: {exc}"
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError,
                json.JSONDecodeError, OSError) as exc:
            last = f"{type(exc).__name__}: {exc}"
    raise RuntimeError(last)
