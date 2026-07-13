#!/usr/bin/env python3
"""Validate the content-addressed prompt profile used by paper execution."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from guardian.prompts import get_prompt_template

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_SCHEMA_PATH = REPO_ROOT / "schemas" / "paper_prompt_profile.schema.json"
EXPECTED_TASKS = {
    "a_box_repair": ("prompt_dev_v4_spec_only", "reasoning_floor_a_box_zero_shot"),
    "t_box_repair": (
        "prompt_dev_v5_tbox_taxonomy_patch",
        "reasoning_floor_t_box_taxonomy_patch_zero_shot",
    ),
}


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_template_sha256(template_name: str) -> str:
    template = get_prompt_template(template_name)
    return canonical_sha256(
        {
            "name": template.name,
            "description": template.description,
            "system_prompt": template.system_prompt,
            "user_prompt_template": template.user_prompt_template,
            "response_format": template.response_format_copy(),
        }
    )


def _resolve_repo_path(path: str) -> Path:
    resolved = (REPO_ROOT / path).resolve()
    if not resolved.is_relative_to(REPO_ROOT):
        raise ValueError(f"Prompt-profile artifact escapes the repository: {path}")
    return resolved


def validate_prompt_profile(profile: dict[str, Any]) -> None:
    schema = json.loads(PROFILE_SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(profile)

    identity_payload = {key: value for key, value in profile.items() if key != "profile_sha256"}
    actual_profile_sha = canonical_sha256(identity_payload)
    if profile["profile_sha256"] != actual_profile_sha:
        raise ValueError(
            "Prompt profile content hash mismatch: "
            f"expected {profile['profile_sha256']}, computed {actual_profile_sha}."
        )

    for task_name, task in profile["tasks"].items():
        expected_version, expected_template = EXPECTED_TASKS[task_name]
        if (task["task_version"], task["template_name"]) != (expected_version, expected_template):
            raise ValueError(
                f"Unexpected paper task contract for {task_name}: "
                f"{task['task_version']} / {task['template_name']}."
            )
        actual_template_sha = prompt_template_sha256(task["template_name"])
        if task["template_sha256"] != actual_template_sha:
            raise ValueError(
                f"Prompt template hash mismatch for {task_name}: expected {task['template_sha256']}, "
                f"computed {actual_template_sha}."
            )
        schema_path = _resolve_repo_path(task["output_schema"]["path"])
        actual_schema_sha = sha256_file(schema_path)
        if task["output_schema"]["sha256"] != actual_schema_sha:
            raise ValueError(
                f"Output schema hash mismatch for {task_name}: expected {task['output_schema']['sha256']}, "
                f"computed {actual_schema_sha}."
            )

    tbox_prompt = get_prompt_template(EXPECTED_TASKS["t_box_repair"][1]).user_prompt_template
    for repair_op in profile["tbox_operation_profile"]["allowed_repair_ops"]:
        if f'"{repair_op}"' not in tbox_prompt:
            raise ValueError(f"Allowed confirmatory T-box operation is missing from the prompt: {repair_op}.")
    for repair_op in profile["tbox_operation_profile"]["excluded_repair_ops"]:
        if f'"{repair_op}"' in tbox_prompt:
            raise ValueError(f"Excluded confirmatory T-box operation is exposed by the prompt: {repair_op}.")

    for artifact_name, artifact in profile["implementation_artifacts"].items():
        artifact_path = _resolve_repo_path(artifact["path"])
        actual_sha = sha256_file(artifact_path)
        if artifact["sha256"] != actual_sha:
            raise ValueError(
                f"Implementation artifact hash mismatch for {artifact_name}: expected {artifact['sha256']}, "
                f"computed {actual_sha}."
            )


def load_prompt_profile(path: str | Path) -> dict[str, Any]:
    profile_path = Path(path)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    validate_prompt_profile(profile)
    return profile


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a paper prompt profile and its content hashes.")
    parser.add_argument(
        "profile",
        nargs="?",
        default="experiments/paper_prompt_profile_v1.json",
        help="Prompt-profile JSON path.",
    )
    args = parser.parse_args()
    profile = load_prompt_profile(args.profile)
    print(
        json.dumps(
            {
                "valid": True,
                "profile_id": profile["profile_id"],
                "status": profile["status"],
                "profile_sha256": profile["profile_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
