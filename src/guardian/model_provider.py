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


def _default_response_format(response_format: dict[str, Any]) -> str | None:
    if response_format.get("type") == "json_object":
        return "json"
    return None


def _env_float(name: str) -> float | None:
    raw = os.getenv(name)
    if raw in (None, ""):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _estimate_cost_usd(
    *,
    provider_env_prefix: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
) -> tuple[float | None, dict[str, float | None]]:
    input_rate = _env_float(f"{provider_env_prefix}_INPUT_COST_PER_1M_TOKENS")
    output_rate = _env_float(f"{provider_env_prefix}_OUTPUT_COST_PER_1M_TOKENS")
    if input_rate is None and output_rate is None:
        return None, {"input_cost_per_1m_tokens_usd": None, "output_cost_per_1m_tokens_usd": None}

    estimated_cost = 0.0
    has_component = False
    if input_rate is not None and isinstance(prompt_tokens, int):
        estimated_cost += (prompt_tokens / 1_000_000) * input_rate
        has_component = True
    if output_rate is not None and isinstance(completion_tokens, int):
        estimated_cost += (completion_tokens / 1_000_000) * output_rate
        has_component = True
    return (estimated_cost if has_component else None), {
        "input_cost_per_1m_tokens_usd": input_rate,
        "output_cost_per_1m_tokens_usd": output_rate,
    }


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
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
        estimated_cost_usd, rate_card = _estimate_cost_usd(
            provider_env_prefix="OPENAI",
            prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
            completion_tokens=completion_tokens if isinstance(completion_tokens, int) else None,
        )
        usage_payload = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": estimated_cost_usd,
        }
        usage_payload.update(rate_card)
        usage_payload["model"] = self.model
        usage_payload["provider"] = "openai"
        usage_payload["request_metadata"] = metadata
        return raw_response, parsed_payload if parsed_payload is not None else text, usage_payload


@dataclass
class OllamaChatProvider:
    api_key: str | None = None
    model: str | None = None
    base_url: str | None = None
    timeout: int = 120

    def __post_init__(self) -> None:
        load_dotenv()
        self.api_key = self.api_key or os.getenv("OLLAMA_API_KEY")
        self.model = self.model or os.getenv("OLLAMA_MODEL")
        self.base_url = (self.base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434/api").rstrip("/")
        if not self.model:
            raise RuntimeError("OLLAMA_MODEL is required for the Ollama provider.")

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        response_format: dict[str, Any],
        metadata: dict[str, Any],
    ) -> tuple[Any, Any, dict[str, Any]]:
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        }
        ollama_format = _default_response_format(response_format)
        if ollama_format:
            payload["format"] = ollama_format
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = requests.post(
            f"{self.base_url}/chat",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        raw_response = response.json()
        message = raw_response.get("message", {})
        text = message.get("content") if isinstance(message, dict) else ""
        text = text or ""
        parsed_payload = _extract_json_payload(text)
        prompt_tokens = raw_response.get("prompt_eval_count")
        completion_tokens = raw_response.get("eval_count")
        total_tokens = None
        if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
            total_tokens = prompt_tokens + completion_tokens
        estimated_cost_usd, rate_card = _estimate_cost_usd(
            provider_env_prefix="OLLAMA",
            prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
            completion_tokens=completion_tokens if isinstance(completion_tokens, int) else None,
        )
        usage_payload = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": estimated_cost_usd,
            "model": self.model,
            "provider": "ollama",
            "request_metadata": metadata,
        }
        usage_payload.update(rate_card)
        return raw_response, parsed_payload if parsed_payload is not None else text, usage_payload


def create_model_provider(model_name: str | None = None) -> ModelProvider:
    load_dotenv()
    provider_name = os.getenv("MODEL_PROVIDER", "openai").strip().lower()
    if provider_name == "openai":
        return OpenAIChatProvider(model=model_name)
    if provider_name == "ollama":
        return OllamaChatProvider(model=model_name)
    raise RuntimeError(f"Unsupported MODEL_PROVIDER: {provider_name}")


@dataclass
class StaticResponseProvider:
    resolver: Callable[[dict[str, Any]], Any]
    provider_name: str = "static"
    model: str = "static-model"

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
            "estimated_cost_usd": metadata.get("estimated_cost_usd"),
            "model": metadata.get("model", self.model),
            "provider": self.provider_name,
        }
        return raw, parsed, usage

