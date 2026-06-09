from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

import requests
from requests import HTTPError, Response

from lib.env import load_dotenv
from lib.utils import iter_jsonl


class ModelProvider(Protocol):
    def generate(
        self,
        prompt: str,
        system_prompt: str,
        response_format: dict[str, Any],
        metadata: dict[str, Any],
    ) -> tuple[Any, Any, dict[str, Any]]:
        ...


@dataclass(frozen=True)
class BatchExecutionResult:
    batch: dict[str, Any]
    output_path: Path | None
    error_path: Path | None


@runtime_checkable
class BatchModelProvider(Protocol):
    def write_batch_request(
        self,
        handle: Any,
        *,
        custom_id: str,
        prompt: str,
        system_prompt: str,
        response_format: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        ...

    def execute_batch(
        self,
        batch_input_path: Path,
        *,
        request_manifest_path: Path,
        output_dir: Path,
        completion_window: str,
        poll_interval_seconds: float,
        status_callback: Callable[[str], None] | None = None,
    ) -> BatchExecutionResult:
        ...

    def parse_batch_result(
        self,
        result_record: dict[str, Any],
        metadata: dict[str, Any],
    ) -> tuple[Any, Any, dict[str, Any], str | None]:
        ...


def _json_candidate_size(candidate: dict[str, Any]) -> int:
    try:
        return len(json.dumps(candidate, ensure_ascii=False, sort_keys=True))
    except (TypeError, ValueError):
        return 0


def _strip_non_json_model_text(raw_text: str) -> str:
    text = re.sub(r"<think\b[^>]*>.*?</think>", "", raw_text, flags=re.IGNORECASE | re.DOTALL).strip()
    if "```" in text:
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return text


def _looks_like_task_payload(candidate: dict[str, Any]) -> bool:
    if not isinstance(candidate.get("case_id"), str):
        return False
    if "predicted_track" in candidate:
        return True
    if "target" in candidate and "ops" in candidate:
        return True
    if "target" in candidate and "proposal" in candidate:
        return True
    if candidate.get("abstain") is True:
        return True
    return False


def _extract_json_payload(raw_text: str) -> Any:
    text = _strip_non_json_model_text(raw_text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            candidate, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            candidates.append(candidate)
    if not candidates:
        return None

    task_payloads = [candidate for candidate in candidates if _looks_like_task_payload(candidate)]
    if task_payloads:
        return max(task_payloads, key=_json_candidate_size)

    with_case_id = [candidate for candidate in candidates if isinstance(candidate.get("case_id"), str)]
    if with_case_id:
        return max(with_case_id, key=_json_candidate_size)

    return max(candidates, key=_json_candidate_size)


def _default_response_format(response_format: dict[str, Any]) -> str | None:
    if response_format.get("type") == "json_object":
        return "json"
    return None


def _openai_message_requests_tools(message: dict[str, Any], finish_reason: Any) -> bool:
    if finish_reason == "tool_calls":
        return True
    if isinstance(message.get("tool_calls"), list) and message.get("tool_calls"):
        return True
    if message.get("function_call") is not None:
        return True
    return False


def _env_float(name: str) -> float | None:
    raw = os.getenv(name)
    if raw in (None, ""):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _env_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw in (None, ""):
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _env_retry_count(name: str) -> int | None:
    raw = os.getenv(name)
    if raw in (None, ""):
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value >= 0 else None


def _ollama_keep_alive_value(value: Any) -> str | int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    normalized = str(value).strip()
    if not normalized:
        return None
    if re.fullmatch(r"\d+", normalized):
        return int(normalized)
    return normalized


def _normalize_api_key(env_name: str, value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if normalized.lower().startswith("bearer "):
        normalized = normalized[7:].strip()
    embedded_prefix = f"{env_name}="
    if normalized.startswith(embedded_prefix):
        normalized = normalized[len(embedded_prefix) :].strip()
    return normalized or None


OPENAI_REASONING_EFFORT_VALUES = frozenset({"none", "minimal", "low", "medium", "high", "xhigh"})


def _normalize_openai_reasoning_effort(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    if normalized not in OPENAI_REASONING_EFFORT_VALUES:
        allowed_values = ", ".join(sorted(OPENAI_REASONING_EFFORT_VALUES))
        raise RuntimeError(f"OPENAI_REASONING_EFFORT must be one of: {allowed_values}.")
    return normalized


def _uses_gpt5_family(model_name: str | None) -> bool:
    if not model_name:
        return False
    normalized = model_name.strip().lower()
    return normalized.startswith("gpt-5")


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


def _metadata_summary(metadata: dict[str, Any]) -> str:
    summary_fields = (
        "run_id",
        "case_id",
        "ablation_bundle",
        "prompt_name",
        "track",
        "task_type",
        "model",
    )
    parts = []
    for field in summary_fields:
        value = metadata.get(field)
        if value is not None:
            parts.append(f"{field}={value!r}")
    return ", ".join(parts) if parts else "no metadata"


def _encode_json_body(payload: dict[str, Any], *, metadata: dict[str, Any], provider_name: str) -> bytes:
    try:
        return json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise RuntimeError(
            f"{provider_name} request payload could not be encoded as strict JSON. "
            f"Request context: {_metadata_summary(metadata)}. Details: {exc}"
        ) from exc


def _openai_chat_payload(
    *,
    model: str,
    prompt: str,
    system_prompt: str,
    response_format: dict[str, Any],
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    }
    if not _uses_gpt5_family(model):
        payload["temperature"] = 0
    if reasoning_effort is not None:
        payload["reasoning"] = {"effort": reasoning_effort}
    if response_format:
        payload["response_format"] = response_format
    return payload


def _openai_message_content(raw_response: dict[str, Any]) -> str:
    choices = raw_response.get("choices", [])
    first_choice = choices[0] if choices else {}
    message = first_choice.get("message", {}) if isinstance(first_choice, dict) else {}
    finish_reason = first_choice.get("finish_reason") if isinstance(first_choice, dict) else None
    if _openai_message_requests_tools(message, finish_reason):
        raise RuntimeError(
            "OpenAI returned a tool-call response even though this client does not configure tools "
            "for chat completions."
        )
    content = message.get("content")
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content or ""


def _openai_usage_payload(
    *,
    raw_response: dict[str, Any],
    metadata: dict[str, Any],
    model: str,
    provider_name: str,
    provider_env_prefix: str = "OPENAI",
) -> dict[str, Any]:
    usage = raw_response.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    prompt_token_details = usage.get("prompt_tokens_details")
    cached_tokens = None
    if isinstance(prompt_token_details, dict):
        cached_value = prompt_token_details.get("cached_tokens")
        if isinstance(cached_value, int):
            cached_tokens = cached_value
    estimated_cost_usd, rate_card = _estimate_cost_usd(
        provider_env_prefix=provider_env_prefix,
        prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
        completion_tokens=completion_tokens if isinstance(completion_tokens, int) else None,
    )
    usage_payload = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": cached_tokens,
        "estimated_cost_usd": estimated_cost_usd,
        "model": raw_response.get("model") if isinstance(raw_response.get("model"), str) else model,
        "provider": provider_name,
        "request_metadata": metadata,
    }
    usage_payload.update(rate_card)
    return usage_payload


def _parse_openai_chat_completion_response(
    raw_response: dict[str, Any],
    *,
    metadata: dict[str, Any],
    model: str,
    provider_name: str,
    provider_env_prefix: str = "OPENAI",
) -> tuple[Any, dict[str, Any]]:
    text = _openai_message_content(raw_response)
    parsed_payload = _extract_json_payload(text)
    usage_payload = _openai_usage_payload(
        raw_response=raw_response,
        metadata=metadata,
        model=model,
        provider_name=provider_name,
        provider_env_prefix=provider_env_prefix,
    )
    return parsed_payload if parsed_payload is not None else text, usage_payload


def _responses_output_text(raw_response: dict[str, Any]) -> str:
    output_text = raw_response.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text
    output = raw_response.get("output")
    parts: list[str] = []
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
    if parts:
        return "\n".join(parts)
    return ""


def _responses_usage_payload(
    *,
    raw_response: dict[str, Any],
    metadata: dict[str, Any],
    model: str,
    provider_name: str,
    provider_env_prefix: str,
) -> dict[str, Any]:
    usage = raw_response.get("usage") or {}
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    total_tokens = usage.get("total_tokens")
    if total_tokens is None and isinstance(input_tokens, int) and isinstance(output_tokens, int):
        total_tokens = input_tokens + output_tokens
    estimated_cost_usd, rate_card = _estimate_cost_usd(
        provider_env_prefix=provider_env_prefix,
        prompt_tokens=input_tokens if isinstance(input_tokens, int) else None,
        completion_tokens=output_tokens if isinstance(output_tokens, int) else None,
    )
    usage_payload = {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": None,
        "estimated_cost_usd": estimated_cost_usd,
        "model": raw_response.get("model") if isinstance(raw_response.get("model"), str) else model,
        "provider": provider_name,
        "request_metadata": metadata,
    }
    usage_payload.update(rate_card)
    return usage_payload


def _write_json_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def _format_duration_compact(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "n/a"
    rounded_seconds = max(0, int(round(seconds)))
    days, remainder = divmod(rounded_seconds, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, secs = divmod(remainder, 60)
    if days:
        return f"{days}d{hours:02d}h"
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _batch_progress_counts(request_counts: dict[str, Any] | None) -> tuple[int | None, int | None]:
    if not isinstance(request_counts, dict):
        return None, None
    total = request_counts.get("total")
    completed = request_counts.get("completed")
    failed = request_counts.get("failed")
    if not isinstance(total, int) or total <= 0:
        return None, None
    done = 0
    if isinstance(completed, int) and completed > 0:
        done += completed
    if isinstance(failed, int) and failed > 0:
        done += failed
    return min(done, total), total


def _estimate_batch_eta_seconds(
    request_counts: dict[str, Any] | None,
    *,
    elapsed_seconds: float | None,
) -> float | None:
    done, total = _batch_progress_counts(request_counts)
    if done is None or total is None:
        return None
    if done >= total:
        return 0.0
    if elapsed_seconds is None or elapsed_seconds <= 0 or done <= 0:
        return None
    remaining = total - done
    return (elapsed_seconds / done) * remaining


def _format_batch_progress(
    request_counts: dict[str, Any] | None,
    *,
    elapsed_seconds: float | None,
) -> str:
    done, total = _batch_progress_counts(request_counts)
    progress_text = f"{done}/{total}" if done is not None and total is not None else "n/a"
    eta_text = _format_duration_compact(
        _estimate_batch_eta_seconds(
            request_counts,
            elapsed_seconds=elapsed_seconds,
        )
    )
    return f"progress={progress_text}; eta={eta_text}; request_counts={request_counts or {}}."


@dataclass
class OpenAIChatProvider:
    api_key: str | None = None
    model: str | None = None
    base_url: str | None = None
    reasoning_effort: str | None = None
    timeout: int = 120
    provider_name: str = "openai"
    provider_env_prefix: str = "OPENAI"
    api_key_env_name: str = "OPENAI_API_KEY"
    model_env_name: str = "OPENAI_MODEL"
    base_url_env_name: str = "OPENAI_BASE_URL"

    def __post_init__(self) -> None:
        load_dotenv()
        self.api_key = _normalize_api_key(self.api_key_env_name, self.api_key or os.getenv(self.api_key_env_name))
        self.model = self.model or os.getenv(self.model_env_name)
        default_base_url = "https://api.openai.com/v1" if self.provider_name == "openai" else ""
        self.base_url = (self.base_url or os.getenv(self.base_url_env_name) or default_base_url).rstrip("/")
        self.reasoning_effort = _normalize_openai_reasoning_effort(
            self.reasoning_effort or (os.getenv("OPENAI_REASONING_EFFORT") if self.provider_name == "openai" else None)
        )
        if not self.api_key:
            raise RuntimeError(f"{self.api_key_env_name} is required for the {self.provider_name} provider.")
        if not self.model:
            raise RuntimeError(f"{self.model_env_name} is required for the {self.provider_name} provider.")
        if not self.base_url:
            raise RuntimeError(f"{self.base_url_env_name} is required for the {self.provider_name} provider.")

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _json_headers(self) -> dict[str, str]:
        headers = self._auth_headers()
        headers["Content-Type"] = "application/json"
        return headers

    def _handle_http_error(self, response: Response, exc: HTTPError, *, action: str) -> None:
        detail = ""
        try:
            error_payload = response.json()
        except ValueError:
            error_payload = {}
        error_message = ((error_payload or {}).get("error") or {}).get("message")
        if error_message:
            detail = f" Details: {error_message}"
        if response.status_code == 401:
            raise RuntimeError(
                f"{self.provider_name} authentication failed (401 Unauthorized). "
                f"Check {self.api_key_env_name} in your shell or .env. "
                f"The value must be the raw API key only, not `{self.api_key_env_name}=...` or `Bearer ...`."
            ) from exc
        if response.status_code == 400:
            raise RuntimeError(
                f"{self.provider_name} {action} failed (400 Bad Request). "
                "Check the configured model and request parameters."
                f"{detail}"
            ) from exc
        raise RuntimeError(f"{self.provider_name} {action} failed ({response.status_code}).{detail}") from exc

    def _download_file(self, file_id: str, destination: Path) -> Path:
        response = requests.get(
            f"{self.base_url}/files/{file_id}/content",
            headers=self._auth_headers(),
            stream=True,
            timeout=self.timeout,
        )
        try:
            response.raise_for_status()
        except HTTPError as exc:
            self._handle_http_error(response, exc, action=f"file download for {file_id}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with open(destination, "wb") as fh:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
        return destination

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        response_format: dict[str, Any],
        metadata: dict[str, Any],
    ) -> tuple[Any, Any, dict[str, Any]]:
        payload = _openai_chat_payload(
            model=self.model,
            prompt=prompt,
            system_prompt=system_prompt,
            response_format=response_format,
            reasoning_effort=self.reasoning_effort,
        )
        request_body = _encode_json_body(payload, metadata=metadata, provider_name="OpenAI")
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._json_headers(),
            data=request_body,
            timeout=self.timeout,
        )
        try:
            response.raise_for_status()
        except HTTPError as exc:
            self._handle_http_error(response, exc, action="request")
        raw_response = response.json()
        parsed_payload, usage_payload = _parse_openai_chat_completion_response(
            raw_response,
            metadata=metadata,
            model=self.model,
            provider_name=self.provider_name,
            provider_env_prefix=self.provider_env_prefix,
        )
        return raw_response, parsed_payload, usage_payload

    def write_batch_request(
        self,
        handle: Any,
        *,
        custom_id: str,
        prompt: str,
        system_prompt: str,
        response_format: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        payload = _openai_chat_payload(
            model=self.model,
            prompt=prompt,
            system_prompt=system_prompt,
            response_format=response_format,
            reasoning_effort=self.reasoning_effort,
        )
        request_record = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": payload,
        }
        handle.write(json.dumps(request_record, ensure_ascii=False) + "\n")
        del metadata

    def execute_batch(
        self,
        batch_input_path: Path,
        *,
        request_manifest_path: Path,
        output_dir: Path,
        completion_window: str,
        poll_interval_seconds: float,
        status_callback: Callable[[str], None] | None = None,
    ) -> BatchExecutionResult:
        del request_manifest_path
        output_dir.mkdir(parents=True, exist_ok=True)
        if status_callback is not None:
            status_callback(f"Uploading batch input file {batch_input_path.name}.")

        with open(batch_input_path, "rb") as input_fh:
            upload_response = requests.post(
                f"{self.base_url}/files",
                headers=self._auth_headers(),
                data={"purpose": "batch"},
                files={"file": (batch_input_path.name, input_fh, "application/jsonl")},
                timeout=self.timeout,
            )
        try:
            upload_response.raise_for_status()
        except HTTPError as exc:
            self._handle_http_error(upload_response, exc, action="batch input upload")
        uploaded_file = upload_response.json()
        _write_json_file(output_dir / "openai_batch_input_file.json", uploaded_file)
        if status_callback is not None:
            status_callback(
                f"Uploaded batch input as {uploaded_file.get('id', 'unknown-file-id')}; creating batch job."
            )

        create_response = requests.post(
            f"{self.base_url}/batches",
            headers=self._json_headers(),
            json={
                "input_file_id": uploaded_file["id"],
                "endpoint": "/v1/chat/completions",
                "completion_window": completion_window,
            },
            timeout=self.timeout,
        )
        try:
            create_response.raise_for_status()
        except HTTPError as exc:
            self._handle_http_error(create_response, exc, action="batch creation")
        batch = create_response.json()
        batch_id = batch.get("id", "unknown-batch-id")
        batch_started_at = time.perf_counter()
        if status_callback is not None:
            status_callback(
                f"Created batch job {batch_id} with status {batch.get('status')!r}; "
                f"polling every {poll_interval_seconds:g}s."
            )

        terminal_statuses = {"completed", "failed", "expired", "cancelled"}
        last_status = batch.get("status")
        last_request_counts = batch.get("request_counts") if isinstance(batch.get("request_counts"), dict) else None
        while batch.get("status") not in terminal_statuses:
            if poll_interval_seconds > 0:
                time.sleep(poll_interval_seconds)
            status_response = requests.get(
                f"{self.base_url}/batches/{batch['id']}",
                headers=self._auth_headers(),
                timeout=self.timeout,
            )
            try:
                status_response.raise_for_status()
            except HTTPError as exc:
                self._handle_http_error(status_response, exc, action=f"batch status lookup for {batch['id']}")
            batch = status_response.json()
            request_counts = batch.get("request_counts") if isinstance(batch.get("request_counts"), dict) else None
            if status_callback is not None and (
                batch.get("status") != last_status or request_counts != last_request_counts
            ):
                elapsed_seconds = time.perf_counter() - batch_started_at
                status_callback(
                    f"Batch {batch.get('id', batch_id)} status {batch.get('status')!r}; "
                    + _format_batch_progress(
                        request_counts,
                        elapsed_seconds=elapsed_seconds,
                    )
                )
            last_status = batch.get("status")
            last_request_counts = dict(request_counts) if isinstance(request_counts, dict) else request_counts

        _write_json_file(output_dir / "openai_batch_job.json", batch)
        if status_callback is not None:
            status_callback(
                f"Batch {batch.get('id', batch_id)} finished with status {batch.get('status')!r}; "
                "downloading result files."
            )

        output_path = None
        output_file_id = batch.get("output_file_id")
        if isinstance(output_file_id, str) and output_file_id:
            output_path = self._download_file(output_file_id, output_dir / "openai_batch_output.jsonl")

        error_path = None
        error_file_id = batch.get("error_file_id")
        if isinstance(error_file_id, str) and error_file_id:
            error_path = self._download_file(error_file_id, output_dir / "openai_batch_errors.jsonl")
        if status_callback is not None:
            status_callback(
                "Downloaded batch artifacts: "
                f"output={'present' if output_path is not None else 'missing'}, "
                f"errors={'present' if error_path is not None else 'missing'}."
            )

        if output_path is None and error_path is None:
            raise RuntimeError(
                f"OpenAI batch {batch.get('id')} ended with status {batch.get('status')!r} "
                "and produced no output or error file."
            )

        return BatchExecutionResult(batch=batch, output_path=output_path, error_path=error_path)

    def parse_batch_result(
        self,
        result_record: dict[str, Any],
        metadata: dict[str, Any],
    ) -> tuple[Any, Any, dict[str, Any], str | None]:
        response_block = result_record.get("response")
        response_body = response_block.get("body") if isinstance(response_block, dict) else None
        status_code = response_block.get("status_code") if isinstance(response_block, dict) else None

        error_message = None
        error_block = result_record.get("error")
        if isinstance(error_block, dict):
            error_message = error_block.get("message") or error_block.get("code")
            if not error_message:
                error_message = json.dumps(error_block, ensure_ascii=False)

        if isinstance(status_code, int) and status_code >= 400:
            body_error = (
                ((response_body.get("error") or {}).get("message"))
                if isinstance(response_body, dict)
                else None
            )
            usage_payload = {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
                "cached_tokens": None,
                "estimated_cost_usd": None,
                "input_cost_per_1m_tokens_usd": _env_float(f"{self.provider_env_prefix}_INPUT_COST_PER_1M_TOKENS"),
                "output_cost_per_1m_tokens_usd": _env_float(f"{self.provider_env_prefix}_OUTPUT_COST_PER_1M_TOKENS"),
                "model": metadata.get("model", self.model),
                "provider": self.provider_name,
                "request_metadata": metadata,
            }
            return (
                response_body if isinstance(response_body, dict) else result_record,
                None,
                usage_payload,
                error_message or body_error or f"Batch request failed with status {status_code}.",
            )

        if isinstance(response_body, dict):
            parsed_payload, usage_payload = _parse_openai_chat_completion_response(
                response_body,
                metadata=metadata,
                model=self.model,
                provider_name=self.provider_name,
                provider_env_prefix=self.provider_env_prefix,
            )
            return response_body, parsed_payload, usage_payload, error_message

        usage_payload = {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "cached_tokens": None,
            "estimated_cost_usd": None,
            "input_cost_per_1m_tokens_usd": _env_float(f"{self.provider_env_prefix}_INPUT_COST_PER_1M_TOKENS"),
            "output_cost_per_1m_tokens_usd": _env_float(f"{self.provider_env_prefix}_OUTPUT_COST_PER_1M_TOKENS"),
            "model": metadata.get("model", self.model),
            "provider": self.provider_name,
            "request_metadata": metadata,
        }
        if error_message is None:
            error_message = "Batch result did not contain a response body."
        return result_record, None, usage_payload, error_message


@dataclass
class OllamaChatProvider:
    api_key: str | None = None
    model: str | None = None
    base_url: str | None = None
    timeout: int = 120
    keep_alive: str | int | None = None
    context_length: int | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    max_retries: int = 2
    retry_base_seconds: float = 2.0
    retry_max_seconds: float = 20.0
    provider_name: str = "ollama"

    def __post_init__(self) -> None:
        load_dotenv()
        self.api_key = _normalize_api_key("OLLAMA_API_KEY", self.api_key or os.getenv("OLLAMA_API_KEY"))
        self.model = self.model or os.getenv("OLLAMA_MODEL")
        self.base_url = (self.base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434/api").rstrip("/")
        self.timeout = _env_int("OLLAMA_TIMEOUT_SECONDS") or self.timeout
        self.keep_alive = _ollama_keep_alive_value(self.keep_alive or os.getenv("OLLAMA_KEEP_ALIVE"))
        self.context_length = self.context_length or _env_int("OLLAMA_CONTEXT_LENGTH")
        self.max_output_tokens = self.max_output_tokens or _env_int("OLLAMA_MAX_OUTPUT_TOKENS")
        self.temperature = self.temperature if self.temperature is not None else _env_float("OLLAMA_TEMPERATURE")
        self.top_p = self.top_p if self.top_p is not None else _env_float("OLLAMA_TOP_P")
        self.seed = self.seed or _env_int("OLLAMA_SEED")
        env_max_retries = _env_retry_count("OLLAMA_MAX_RETRIES")
        self.max_retries = env_max_retries if env_max_retries is not None else self.max_retries
        self.retry_base_seconds = _env_float("OLLAMA_RETRY_BASE_SECONDS") or self.retry_base_seconds
        self.retry_max_seconds = _env_float("OLLAMA_RETRY_MAX_SECONDS") or self.retry_max_seconds
        if not self.model:
            raise RuntimeError("OLLAMA_MODEL is required for the Ollama provider.")
        if self.retry_base_seconds <= 0:
            self.retry_base_seconds = 2.0
        if self.retry_max_seconds <= 0:
            self.retry_max_seconds = self.retry_base_seconds

    def _retry_delay_seconds(self, retry_index: int) -> float:
        exponential_delay = self.retry_base_seconds * (2 ** max(0, retry_index - 1))
        capped_delay = min(exponential_delay, self.retry_max_seconds)
        return min(self.retry_max_seconds, capped_delay + random.uniform(0, min(1.0, self.retry_base_seconds)))

    def _post_chat(self, *, headers: dict[str, str], payload: dict[str, Any]) -> Response:
        retryable_statuses = {429, 500, 502, 503, 504}
        retryable_exceptions = (requests.Timeout, requests.ConnectionError)
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    f"{self.base_url}/chat",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
            except retryable_exceptions as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        f"Ollama chat request failed after {attempt + 1} attempt(s): {exc}"
                    ) from exc
            else:
                if response.status_code not in retryable_statuses or attempt >= self.max_retries:
                    return response
                last_error = RuntimeError(f"retryable HTTP {response.status_code}")
            time.sleep(self._retry_delay_seconds(attempt + 1))
        if last_error is not None:
            raise RuntimeError(
                f"Ollama chat request failed after {self.max_retries + 1} attempt(s): {last_error}"
            ) from last_error
        raise RuntimeError("Ollama chat request failed before a response was returned.")

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
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        options: dict[str, Any] = {}
        if isinstance(self.context_length, int) and self.context_length > 0:
            options["num_ctx"] = self.context_length
        if isinstance(self.max_output_tokens, int) and self.max_output_tokens > 0:
            options["num_predict"] = self.max_output_tokens
        if self.temperature is not None:
            options["temperature"] = self.temperature
        if self.top_p is not None:
            options["top_p"] = self.top_p
        if isinstance(self.seed, int):
            options["seed"] = self.seed
        if options:
            payload["options"] = options
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = self._post_chat(headers=headers, payload=payload)
        try:
            response.raise_for_status()
        except HTTPError as exc:
            if response.status_code == 401:
                raise RuntimeError(
                    "Ollama authentication failed (401 Unauthorized). "
                    "Check OLLAMA_API_KEY in your shell or .env. "
                    "The value must be the raw API key only, not `OLLAMA_API_KEY=...` or `Bearer ...`."
                ) from exc
            raise RuntimeError(f"Ollama chat request failed ({response.status_code}).") from exc
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
            "cached_tokens": None,
            "estimated_cost_usd": estimated_cost_usd,
            "model": self.model,
            "provider": self.provider_name,
            "request_metadata": metadata,
        }
        usage_payload.update(rate_card)
        return raw_response, parsed_payload if parsed_payload is not None else text, usage_payload


@dataclass
class OpenAIResponsesProvider:
    api_key: str | None = None
    model: str | None = None
    base_url: str | None = None
    timeout: int = 120
    max_output_tokens: int | None = None
    max_retries: int = 4
    retry_base_seconds: float = 2.0
    retry_max_seconds: float = 20.0
    provider_name: str = "university"
    provider_env_prefix: str = "UNIVERSITY_OPENAI"

    def __post_init__(self) -> None:
        load_dotenv()
        self.api_key = _normalize_api_key(
            f"{self.provider_env_prefix}_API_KEY",
            self.api_key or os.getenv(f"{self.provider_env_prefix}_API_KEY"),
        )
        self.model = self.model or os.getenv(f"{self.provider_env_prefix}_MODEL")
        self.base_url = (self.base_url or os.getenv(f"{self.provider_env_prefix}_BASE_URL") or "").rstrip("/")
        self.timeout = _env_int(f"{self.provider_env_prefix}_TIMEOUT_SECONDS") or self.timeout
        self.max_output_tokens = self.max_output_tokens or _env_int(f"{self.provider_env_prefix}_MAX_OUTPUT_TOKENS")
        env_max_retries = _env_retry_count(f"{self.provider_env_prefix}_MAX_RETRIES")
        self.max_retries = env_max_retries if env_max_retries is not None else self.max_retries
        self.retry_base_seconds = (
            _env_float(f"{self.provider_env_prefix}_RETRY_BASE_SECONDS") or self.retry_base_seconds
        )
        self.retry_max_seconds = _env_float(f"{self.provider_env_prefix}_RETRY_MAX_SECONDS") or self.retry_max_seconds
        if not self.api_key:
            raise RuntimeError(f"{self.provider_env_prefix}_API_KEY is required for the {self.provider_name} provider.")
        if not self.model:
            raise RuntimeError(f"{self.provider_env_prefix}_MODEL is required for the {self.provider_name} provider.")
        if not self.base_url:
            raise RuntimeError(
                f"{self.provider_env_prefix}_BASE_URL is required for the {self.provider_name} provider."
            )
        if self.retry_base_seconds <= 0:
            self.retry_base_seconds = 2.0
        if self.retry_max_seconds <= 0:
            self.retry_max_seconds = self.retry_base_seconds

    def _retry_delay_seconds(self, retry_index: int) -> float:
        exponential_delay = self.retry_base_seconds * (2 ** max(0, retry_index - 1))
        capped_delay = min(exponential_delay, self.retry_max_seconds)
        return min(self.retry_max_seconds, capped_delay + random.uniform(0, min(1.0, self.retry_base_seconds)))

    def _json_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _handle_http_error(self, response: Response, exc: HTTPError) -> None:
        detail = ""
        try:
            error_payload = response.json()
        except ValueError:
            error_payload = {}
        error = error_payload.get("error") if isinstance(error_payload, dict) else None
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            detail = f" Details: {error['message']}"
        if response.status_code == 401:
            raise RuntimeError(
                f"{self.provider_name} authentication failed (401 Unauthorized). "
                f"Check {self.provider_env_prefix}_API_KEY in your shell or .env."
            ) from exc
        raise RuntimeError(f"{self.provider_name} Responses request failed ({response.status_code}).{detail}") from exc

    def _post_responses(self, request_body: bytes) -> Response:
        retryable_statuses = {429, 500, 502, 503, 504}
        retryable_exceptions = (requests.Timeout, requests.ConnectionError)
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    f"{self.base_url}/responses",
                    headers=self._json_headers(),
                    data=request_body,
                    timeout=self.timeout,
                )
            except retryable_exceptions as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        f"{self.provider_name} Responses request failed after {attempt + 1} attempt(s): {exc}"
                    ) from exc
            else:
                if response.status_code not in retryable_statuses or attempt >= self.max_retries:
                    return response
                last_error = RuntimeError(f"retryable HTTP {response.status_code}")
            time.sleep(self._retry_delay_seconds(attempt + 1))
        if last_error is not None:
            raise RuntimeError(
                f"{self.provider_name} Responses request failed after {self.max_retries + 1} attempt(s): {last_error}"
            ) from last_error
        raise RuntimeError(f"{self.provider_name} Responses request failed before a response was returned.")

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        response_format: dict[str, Any],
        metadata: dict[str, Any],
    ) -> tuple[Any, Any, dict[str, Any]]:
        payload: dict[str, Any] = {
            "model": self.model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        }
        if response_format.get("type") == "json_object":
            payload["text"] = {"format": {"type": "json_object"}}
        if isinstance(self.max_output_tokens, int) and self.max_output_tokens > 0:
            payload["max_output_tokens"] = self.max_output_tokens
        request_body = _encode_json_body(payload, metadata=metadata, provider_name=self.provider_name)
        response = self._post_responses(request_body)
        try:
            response.raise_for_status()
        except HTTPError as exc:
            self._handle_http_error(response, exc)
        raw_response = response.json()
        text = _responses_output_text(raw_response)
        parsed_payload = _extract_json_payload(text)
        usage_payload = _responses_usage_payload(
            raw_response=raw_response,
            metadata=metadata,
            model=self.model,
            provider_name=self.provider_name,
            provider_env_prefix=self.provider_env_prefix,
        )
        return raw_response, parsed_payload if parsed_payload is not None else text, usage_payload


def create_model_provider(model_name: str | None = None, model_endpoint: str | None = None) -> ModelProvider:
    load_dotenv()
    provider_name = (
        model_endpoint or os.getenv("MODEL_ENDPOINT") or os.getenv("MODEL_PROVIDER") or "openai"
    ).strip().lower()
    if provider_name == "openai":
        return OpenAIChatProvider(model=model_name)
    if provider_name == "ollama":
        return OllamaChatProvider(model=model_name)
    if provider_name == "azure":
        return OpenAIChatProvider(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            model=model_name or os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
            provider_name="azure",
            provider_env_prefix="AZURE_OPENAI",
            api_key_env_name="AZURE_OPENAI_API_KEY",
            model_env_name="AZURE_OPENAI_DEPLOYMENT",
            base_url_env_name="AZURE_OPENAI_ENDPOINT",
        )
    if provider_name == "university":
        return OpenAIResponsesProvider(model=model_name)
    raise RuntimeError(f"Unsupported model endpoint: {provider_name}")


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
            "cached_tokens": metadata.get("cached_tokens"),
            "estimated_cost_usd": metadata.get("estimated_cost_usd"),
            "model": metadata.get("model", self.model),
            "provider": self.provider_name,
        }
        return raw, parsed, usage

    def write_batch_request(
        self,
        handle: Any,
        *,
        custom_id: str,
        prompt: str,
        system_prompt: str,
        response_format: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        del metadata
        handle.write(
            json.dumps(
                {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                        "response_format": response_format,
                    },
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    def execute_batch(
        self,
        batch_input_path: Path,
        *,
        request_manifest_path: Path,
        output_dir: Path,
        completion_window: str,
        poll_interval_seconds: float,
        status_callback: Callable[[str], None] | None = None,
    ) -> BatchExecutionResult:
        del completion_window, poll_interval_seconds
        output_dir.mkdir(parents=True, exist_ok=True)
        if status_callback is not None:
            status_callback(f"Processing static batch input file {batch_input_path.name}.")
        manifest_by_custom_id = {}
        for row in iter_jsonl(request_manifest_path):
            if not isinstance(row, dict):
                continue
            custom_id = row.get("custom_id")
            if isinstance(custom_id, str) and custom_id:
                manifest_by_custom_id[custom_id] = row

        output_path = output_dir / "static_batch_output.jsonl"
        error_path = output_dir / "static_batch_errors.jsonl"
        has_errors = False
        completed = 0
        total = 0
        with open(output_path, "w", encoding="utf-8") as output_fh, open(error_path, "w", encoding="utf-8") as error_fh:
            for request_row in iter_jsonl(batch_input_path):
                if not isinstance(request_row, dict):
                    continue
                total += 1
                custom_id = request_row.get("custom_id")
                if not isinstance(custom_id, str) or not custom_id:
                    has_errors = True
                    error_fh.write(
                        json.dumps(
                            {"error": {"message": "Batch request is missing a custom_id."}},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    continue
                metadata = dict((manifest_by_custom_id.get(custom_id) or {}).get("metadata") or {})
                try:
                    payload = self.resolver(dict(metadata))
                    response_body = {
                        "id": f"chatcmpl-{custom_id}",
                        "object": "chat.completion",
                        "model": metadata.get("model", self.model),
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": json.dumps(payload, ensure_ascii=False),
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": metadata.get("prompt_tokens", 0),
                            "completion_tokens": metadata.get("completion_tokens", 0),
                            "total_tokens": metadata.get("total_tokens", 0),
                        },
                    }
                    output_fh.write(
                        json.dumps(
                            {
                                "custom_id": custom_id,
                                "response": {"status_code": 200, "body": response_body},
                                "error": None,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    completed += 1
                except Exception as exc:
                    has_errors = True
                    error_fh.write(
                        json.dumps(
                            {
                                "custom_id": custom_id,
                                "response": None,
                                "error": {"message": str(exc)},
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

        if not has_errors:
            error_path.unlink()
            error_path_result = None
        else:
            error_path_result = error_path

        batch_payload = {
            "id": "static-batch",
            "status": "completed",
            "request_counts": {
                "total": total,
                "completed": completed,
                "failed": total - completed,
            },
        }
        _write_json_file(output_dir / "static_batch_job.json", batch_payload)
        if status_callback is not None:
            status_callback(
                f"Static batch completed with {completed}/{total} successful requests."
            )
        return BatchExecutionResult(batch=batch_payload, output_path=output_path, error_path=error_path_result)

    def parse_batch_result(
        self,
        result_record: dict[str, Any],
        metadata: dict[str, Any],
    ) -> tuple[Any, Any, dict[str, Any], str | None]:
        response_block = result_record.get("response")
        response_body = response_block.get("body") if isinstance(response_block, dict) else None
        error_block = result_record.get("error")
        error_message = None
        if isinstance(error_block, dict):
            error_message = error_block.get("message") or error_block.get("code")
            if not error_message:
                error_message = json.dumps(error_block, ensure_ascii=False)

        prompt_tokens = metadata.get("prompt_tokens", 0)
        completion_tokens = metadata.get("completion_tokens", 0)
        total_tokens = metadata.get("total_tokens", 0)
        usage_payload = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cached_tokens": metadata.get("cached_tokens"),
            "estimated_cost_usd": metadata.get("estimated_cost_usd"),
            "model": metadata.get("model", self.model),
            "provider": self.provider_name,
            "request_metadata": metadata,
        }

        if not isinstance(response_body, dict):
            if error_message is None:
                error_message = "Batch result did not contain a response body."
            return result_record, None, usage_payload, error_message

        text = _openai_message_content(response_body)
        parsed_payload = _extract_json_payload(text)
        return response_body, parsed_payload if parsed_payload is not None else text, usage_payload, error_message
