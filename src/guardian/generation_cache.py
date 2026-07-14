"""Content-addressed model-generation cache shared across experiment runs."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from guardian.model_provider import BatchExecutionResult, BatchModelProvider, ModelProvider

CACHE_SCHEMA_VERSION = 1
REQUEST_SPEC_VERSION = 1


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def generation_request_key(specification: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(specification).encode("utf-8")).hexdigest()


def build_generation_request_spec(
    *,
    provider: str,
    model: str,
    model_digest: str,
    inference_settings: dict[str, Any],
    prompt: str,
    system_prompt: str,
    response_format: dict[str, Any],
) -> dict[str, Any]:
    """Describe every input that can change a provider generation."""
    return {
        "request_spec_version": REQUEST_SPEC_VERSION,
        "provider": provider,
        "model": model,
        "model_digest": model_digest,
        "inference_settings": inference_settings,
        "system_prompt": system_prompt,
        "prompt": prompt,
        "response_format": response_format,
    }


class GenerationCache:
    """Append-only SQLite store keyed by the canonical generation request."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=60.0)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=60000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS generations (
                    request_key TEXT PRIMARY KEY,
                    request_spec_json TEXT NOT NULL,
                    raw_response_json TEXT NOT NULL,
                    parsed_payload_json TEXT NOT NULL,
                    usage_json TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "INSERT OR IGNORE INTO cache_metadata(key, value) VALUES('schema_version', ?)",
                (str(CACHE_SCHEMA_VERSION),),
            )
            row = connection.execute(
                "SELECT value FROM cache_metadata WHERE key='schema_version'"
            ).fetchone()
            if row is None or row[0] != str(CACHE_SCHEMA_VERSION):
                raise RuntimeError(
                    f"Unsupported generation cache schema at {self.path}: {row[0] if row else None}."
                )

    def get(self, specification: dict[str, Any]) -> dict[str, Any] | None:
        request_key = generation_request_key(specification)
        expected_spec = _canonical_json(specification)
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT request_spec_json, raw_response_json, parsed_payload_json, usage_json,
                       created_at_utc
                FROM generations WHERE request_key=?
                """,
                (request_key,),
            ).fetchone()
        if row is None:
            return None
        if row[0] != expected_spec:
            raise RuntimeError(f"Generation-cache key collision for {request_key}.")
        return {
            "request_key": request_key,
            "raw_response": json.loads(row[1]),
            "parsed_payload": json.loads(row[2]),
            "usage": json.loads(row[3]),
            "created_at_utc": row[4],
        }

    def existing_keys(self, request_keys: list[str] | set[str]) -> set[str]:
        """Return cached keys without opening one SQLite connection per planned request."""
        normalized = sorted({key for key in request_keys if isinstance(key, str) and key})
        if not normalized:
            return set()
        found: set[str] = set()
        with self._lock, self._connect() as connection:
            for offset in range(0, len(normalized), 500):
                batch = normalized[offset : offset + 500]
                placeholders = ",".join("?" for _ in batch)
                rows = connection.execute(
                    f"SELECT request_key FROM generations WHERE request_key IN ({placeholders})",  # noqa: S608
                    batch,
                )
                found.update(str(row[0]) for row in rows)
        return found

    def put(
        self,
        specification: dict[str, Any],
        *,
        raw_response: Any,
        parsed_payload: Any,
        usage: dict[str, Any],
    ) -> bool:
        request_key = generation_request_key(specification)
        values = (
            request_key,
            _canonical_json(specification),
            _canonical_json(raw_response),
            _canonical_json(parsed_payload),
            _canonical_json(usage),
            datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        )
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO generations(
                    request_key, request_spec_json, raw_response_json, parsed_payload_json,
                    usage_json, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                values,
            )
            inserted = cursor.rowcount == 1
            if not inserted:
                existing = connection.execute(
                    "SELECT request_spec_json FROM generations WHERE request_key=?", (request_key,)
                ).fetchone()
                if existing is None or existing[0] != values[1]:
                    raise RuntimeError(f"Generation-cache key collision for {request_key}.")
        return inserted


def _cache_hit_usage(
    original: dict[str, Any], *, metadata: dict[str, Any], request_key: str
) -> dict[str, Any]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
        "estimated_cost_usd": 0.0,
        "input_cost_per_1m_tokens_usd": original.get("input_cost_per_1m_tokens_usd"),
        "output_cost_per_1m_tokens_usd": original.get("output_cost_per_1m_tokens_usd"),
        "model": original.get("model") or metadata.get("model"),
        "provider": original.get("provider"),
        "request_metadata": metadata,
        "generation_cache_hit": True,
        "generation_request_key": request_key,
        "source_generation_usage": original,
    }


class CachedModelProvider:
    """Provider decorator that reuses exact generations across runs and populations."""

    def __init__(
        self,
        delegate: ModelProvider,
        *,
        cache_path: str | Path,
        model_digest: str,
        inference_settings: dict[str, Any],
    ):
        if not model_digest.strip():
            raise ValueError("A stable model digest is required when generation caching is enabled.")
        self.delegate = delegate
        self.cache = GenerationCache(cache_path)
        self.cache_path = self.cache.path
        self.model_digest = model_digest.strip()
        self.inference_settings = inference_settings
        self.provider_name = str(getattr(delegate, "provider_name", delegate.__class__.__name__))
        self.model = str(getattr(delegate, "model", "unknown-model"))
        self.reasoning_effort = getattr(delegate, "reasoning_effort", None)
        self.stats = {"hits": 0, "misses": 0, "stores": 0}
        self._stats_lock = threading.Lock()
        self._request_specs: dict[str, dict[str, Any]] = {}
        self._pending_cached_rows: dict[str, dict[str, Any]] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    def _increment(self, field: str) -> None:
        with self._stats_lock:
            self.stats[field] += 1

    def _specification(
        self,
        *,
        prompt: str,
        system_prompt: str,
        response_format: dict[str, Any],
    ) -> dict[str, Any]:
        return build_generation_request_spec(
            provider=self.provider_name,
            model=self.model,
            model_digest=self.model_digest,
            inference_settings=self.inference_settings,
            prompt=prompt,
            system_prompt=system_prompt,
            response_format=response_format,
        )

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        response_format: dict[str, Any],
        metadata: dict[str, Any],
    ) -> tuple[Any, Any, dict[str, Any]]:
        specification = self._specification(
            prompt=prompt, system_prompt=system_prompt, response_format=response_format
        )
        cached = self.cache.get(specification)
        if cached is not None:
            self._increment("hits")
            return (
                cached["raw_response"],
                cached["parsed_payload"],
                _cache_hit_usage(
                    cached["usage"], metadata=metadata, request_key=cached["request_key"]
                ),
            )
        self._increment("misses")
        raw_response, parsed_payload, usage = self.delegate.generate(
            prompt, system_prompt, response_format, metadata
        )
        if self.cache.put(
            specification,
            raw_response=raw_response,
            parsed_payload=parsed_payload,
            usage=usage,
        ):
            self._increment("stores")
        return raw_response, parsed_payload, usage

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
        if not isinstance(self.delegate, BatchModelProvider):
            raise RuntimeError(f"{self.delegate.__class__.__name__} does not support batch execution.")
        specification = self._specification(
            prompt=prompt, system_prompt=system_prompt, response_format=response_format
        )
        self._request_specs[custom_id] = specification
        cached = self.cache.get(specification)
        if cached is None:
            self._increment("misses")
            self.delegate.write_batch_request(
                handle,
                custom_id=custom_id,
                prompt=prompt,
                system_prompt=system_prompt,
                response_format=response_format,
                metadata=metadata,
            )
            return
        self._increment("hits")
        self._pending_cached_rows[custom_id] = {
            "custom_id": custom_id,
            "response": {"status_code": 200, "body": cached["raw_response"]},
            "error": None,
            "generation_cache": {
                "hit": True,
                "request_key": cached["request_key"],
                "source_usage": cached["usage"],
            },
        }

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
        if not isinstance(self.delegate, BatchModelProvider):
            raise RuntimeError(f"{self.delegate.__class__.__name__} does not support batch execution.")
        has_provider_requests = batch_input_path.is_file() and batch_input_path.stat().st_size > 0
        delegate_result: BatchExecutionResult | None = None
        if has_provider_requests:
            delegate_result = self.delegate.execute_batch(
                batch_input_path,
                request_manifest_path=request_manifest_path,
                output_dir=output_dir,
                completion_window=completion_window,
                poll_interval_seconds=poll_interval_seconds,
                status_callback=status_callback,
            )
        cached_rows = list(self._pending_cached_rows.values())
        self._pending_cached_rows.clear()
        if not cached_rows:
            if delegate_result is None:
                raise RuntimeError("Generation-cache batch contained neither hits nor provider requests.")
            return delegate_result
        output_dir.mkdir(parents=True, exist_ok=True)
        merged_output = output_dir / "generation_cache_merged_batch_output.jsonl"
        with merged_output.open("w", encoding="utf-8") as handle:
            for row in cached_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if delegate_result is not None and delegate_result.output_path is not None:
                with delegate_result.output_path.open("r", encoding="utf-8") as source:
                    shutil.copyfileobj(source, handle)
        provider_counts = {"total": 0, "completed": 0, "failed": 0}
        if delegate_result is not None:
            counts = delegate_result.batch.get("request_counts")
            if isinstance(counts, dict):
                for key in provider_counts:
                    if isinstance(counts.get(key), int):
                        provider_counts[key] = counts[key]
        batch = dict(delegate_result.batch) if delegate_result is not None else {}
        batch.update(
            {
                "id": batch.get("id") or "generation-cache-only",
                "status": batch.get("status") or "completed",
                "generation_cache_hits": len(cached_rows),
                "provider_request_count": provider_counts["total"],
                "request_counts": {
                    "total": provider_counts["total"] + len(cached_rows),
                    "completed": provider_counts["completed"] + len(cached_rows),
                    "failed": provider_counts["failed"],
                },
            }
        )
        return BatchExecutionResult(
            batch=batch,
            output_path=merged_output,
            error_path=delegate_result.error_path if delegate_result is not None else None,
        )

    def parse_batch_result(
        self,
        result_record: dict[str, Any],
        metadata: dict[str, Any],
    ) -> tuple[Any, Any, dict[str, Any], str | None]:
        if not isinstance(self.delegate, BatchModelProvider):
            raise RuntimeError(f"{self.delegate.__class__.__name__} does not support batch execution.")
        custom_id = result_record.get("custom_id")
        raw_response, parsed_payload, usage, error_message = self.delegate.parse_batch_result(
            result_record, metadata
        )
        cache_details = result_record.get("generation_cache")
        if isinstance(cache_details, dict) and cache_details.get("hit") is True:
            request_key = str(cache_details.get("request_key") or "")
            original_usage = cache_details.get("source_usage")
            return (
                raw_response,
                parsed_payload,
                _cache_hit_usage(
                    original_usage if isinstance(original_usage, dict) else usage,
                    metadata=metadata,
                    request_key=request_key,
                ),
                error_message,
            )
        if (
            error_message is None
            and raw_response is not None
            and isinstance(custom_id, str)
            and custom_id in self._request_specs
        ):
            if self.cache.put(
                self._request_specs[custom_id],
                raw_response=raw_response,
                parsed_payload=parsed_payload,
                usage=usage,
            ):
                self._increment("stores")
        return raw_response, parsed_payload, usage, error_message
