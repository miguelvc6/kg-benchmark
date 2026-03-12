from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import requests

from lib.env import load_dotenv


class ModelProvider(Protocol):
    def generate(
        self,
        prompt: str,
        system_prompt: str,
        response_format: dict[str, Any],
        metadata: dict[str, Any],
    ) -> tuple[Any, Any, dict[str, Any]]:
        ...


def _extract_json_payload(raw_text: str) -> Any:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


@dataclass
class OpenAIChatProvider:
    api_key: str | None = None
    model: str | None = None
    base_url: str | None = None
    timeout: int = 120

    def __post_init__(self) -> None:
        load_dotenv()
        self.api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        self.model = self.model or os.getenv("OPENAI_MODEL")
        self.base_url = (self.base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for the OpenAI provider.")
        if not self.model:
            raise RuntimeError("OPENAI_MODEL is required for the OpenAI provider.")

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        response_format: dict[str, Any],
        metadata: dict[str, Any],
    ) -> tuple[Any, Any, dict[str, Any]]:
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        }
        if response_format:
            payload["response_format"] = response_format
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        raw_response = response.json()
        choices = raw_response.get("choices", [])
        message = choices[0].get("message", {}) if choices else {}
        content = message.get("content")
        if isinstance(content, list):
            text = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        else:
            text = content or ""
        parsed_payload = _extract_json_payload(text)
        usage = raw_response.get("usage") or {}
        usage_payload = {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }
        usage_payload["model"] = self.model
        usage_payload["provider"] = "openai"
        usage_payload["request_metadata"] = metadata
        return raw_response, parsed_payload if parsed_payload is not None else text, usage_payload


@dataclass
class StaticResponseProvider:
    resolver: Callable[[dict[str, Any]], Any]
    provider_name: str = "static"

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        response_format: dict[str, Any],
        metadata: dict[str, Any],
    ) -> tuple[Any, Any, dict[str, Any]]:
        del prompt, system_prompt, response_format
        payload = self.resolver(dict(metadata))
        raw = payload
        parsed = payload
        usage = {
            "prompt_tokens": metadata.get("prompt_tokens", 0),
            "completion_tokens": metadata.get("completion_tokens", 0),
            "total_tokens": metadata.get("total_tokens", 0),
            "model": metadata.get("model", "static-model"),
            "provider": self.provider_name,
        }
        return raw, parsed, usage

