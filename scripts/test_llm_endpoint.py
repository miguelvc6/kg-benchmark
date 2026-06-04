from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from openai import OpenAI

from lib.env import load_dotenv

DEFAULT_PROMPT = "What is the capital of France?"


@dataclass(frozen=True)
class EndpointConfig:
    name: str
    model: str
    base_url: str
    api_key: str | None = None


def _is_missing(value: str | None) -> bool:
    if value is None:
        return True
    stripped = value.strip()
    if not stripped:
        return True
    lowered = stripped.lower()
    return lowered in {
        "your_api_key",
        "your-api-key",
        "replace_me",
        "replace-with-your-key",
        "add-api-key",
        "<your-api-key>",
        "<your_api_key>",
        "todo",
    }


def _require_env(name: str, *, allow_placeholder: bool = False) -> str:
    value = os.getenv(name)
    if _is_missing(value) and not allow_placeholder:
        raise RuntimeError(f"{name} is missing or still set to a placeholder value.")
    return (value or "").strip()


def _optional_env(name: str) -> str | None:
    value = os.getenv(name)
    if _is_missing(value):
        return None
    return value.strip()


def _ollama_config() -> EndpointConfig:
    return EndpointConfig(
        name="ollama",
        model=_require_env("OLLAMA_MODEL"),
        base_url=(os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434/api").strip().rstrip("/"),
        api_key=_optional_env("OLLAMA_API_KEY"),
    )


def _azure_config() -> EndpointConfig:
    return EndpointConfig(
        name="azure",
        model=_require_env("AZURE_OPENAI_DEPLOYMENT"),
        base_url=_require_env("AZURE_OPENAI_ENDPOINT").rstrip("/"),
        api_key=_require_env("AZURE_OPENAI_API_KEY"),
    )


def _university_config() -> EndpointConfig:
    return EndpointConfig(
        name="university",
        model=_require_env("UNIVERSITY_OPENAI_MODEL"),
        base_url=_require_env("UNIVERSITY_OPENAI_BASE_URL").rstrip("/"),
        api_key=_require_env("UNIVERSITY_OPENAI_API_KEY"),
    )


def _extract_responses_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = getattr(response, "output", None) or []
    parts: list[str] = []
    for item in output:
        for content in getattr(item, "content", None) or []:
            text = getattr(content, "text", None)
            if isinstance(text, str):
                parts.append(text)
    if parts:
        return "\n".join(parts).strip()

    return str(response)


def _test_ollama(prompt: str, timeout: float) -> str:
    config = _ollama_config()
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    payload: dict[str, Any] = {
        "model": config.model,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
    }
    keep_alive = _optional_env("OLLAMA_KEEP_ALIVE")
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive
    context_length = _optional_env("OLLAMA_CONTEXT_LENGTH")
    if context_length is not None:
        try:
            payload["options"] = {"num_ctx": int(context_length)}
        except ValueError as exc:
            raise RuntimeError("OLLAMA_CONTEXT_LENGTH must be an integer.") from exc

    response = requests.post(f"{config.base_url}/chat", headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    message = data.get("message") if isinstance(data, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"Ollama response did not include message.content: {data}")
    return content.strip()


def _test_azure(prompt: str, timeout: float) -> str:
    config = _azure_config()
    client = OpenAI(base_url=config.base_url, api_key=config.api_key, timeout=timeout)
    completion = client.chat.completions.create(
        model=config.model,
        messages=[{"role": "user", "content": prompt}],
    )
    message = completion.choices[0].message
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content.strip()
    return str(message)


def _test_university(prompt: str, timeout: float) -> str:
    config = _university_config()
    client = OpenAI(base_url=config.base_url, api_key=config.api_key, timeout=timeout)
    response = client.responses.create(
        model=config.model,
        input=[{"role": "user", "content": prompt}],
    )
    return _extract_responses_text(response)


TESTERS = {
    "ollama": _test_ollama,
    "azure": _test_azure,
    "university": _test_university,
}


def _run_one(name: str, *, prompt: str, timeout: float) -> bool:
    print(f"\n== {name} ==")
    try:
        answer = TESTERS[name](prompt, timeout)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return False
    print(answer)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test configured LLM inference endpoints.")
    parser.add_argument(
        "approach",
        nargs="?",
        choices=("ollama", "azure", "university", "all"),
        default=None,
        help="Endpoint family to test.",
    )
    parser.add_argument(
        "--model-endpoint",
        choices=("ollama", "azure", "university", "all"),
        default=None,
        help="Endpoint family to test. Equivalent to the positional approach argument.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt to send to the selected endpoint.")
    parser.add_argument("--timeout", type=float, default=60.0, help="Request timeout in seconds.")
    parser.add_argument("--dotenv", default=".env", help="Path to the environment file to load.")
    args = parser.parse_args()

    dotenv_path = Path(args.dotenv)
    if dotenv_path.exists():
        load_dotenv(dotenv_path)
    else:
        load_dotenv()

    approach = args.model_endpoint or args.approach
    if approach is None:
        parser.error("choose an endpoint with the positional approach or --model-endpoint")
    names = tuple(TESTERS) if approach == "all" else (approach,)
    results = [_run_one(name, prompt=args.prompt, timeout=args.timeout) for name in names]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
