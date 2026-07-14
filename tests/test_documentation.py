from __future__ import annotations

import re
import unittest
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
DOC_ROOTS = (ROOT / "docs", ROOT / "docs-conceptual", ROOT / "docs-technical")
WORKFLOW_DOCUMENTS = (ROOT / "README.md", *(ROOT / "docs-technical").glob("*.md"))
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
MACHINE_PATH_PATTERNS = (
    re.compile(r"/mnt/[A-Za-z]/"),
    re.compile(r"/(?:home|Users)/[A-Za-z0-9._-]+/"),
    re.compile(r"\b[A-Za-z]:\\"),
)
CONCRETE_SSH_ENDPOINT = re.compile(
    r"\b[A-Za-z0-9._-]+@(?:\d{1,3}\.){3}\d{1,3}\b"
)


def _link_target(raw_target: str) -> str:
    target = raw_target.strip()
    if target.startswith("<") and ">" in target:
        return target[1 : target.index(">")]
    return target.split(maxsplit=1)[0]


class TechnicalDocumentationTests(unittest.TestCase):
    def _documents(self) -> list[Path]:
        documents = sorted(path for root in DOC_ROOTS for path in root.rglob("*.md"))
        self.assertTrue(documents)
        return documents

    def test_repository_local_markdown_links_resolve(self) -> None:
        failures: list[str] = []
        for document in self._documents():
            text = document.read_text(encoding="utf-8")
            for match in MARKDOWN_LINK.finditer(text):
                target = _link_target(match.group(1))
                parsed = urlsplit(target)
                if parsed.scheme or target.startswith("#"):
                    continue
                path_text = unquote(parsed.path)
                if not path_text:
                    continue
                if Path(path_text).is_absolute():
                    failures.append(f"{document.relative_to(ROOT)}: absolute local link {target}")
                    continue
                resolved = (document.parent / path_text).resolve()
                if not resolved.exists():
                    failures.append(f"{document.relative_to(ROOT)}: missing link target {target}")
        self.assertEqual(failures, [], "\n".join(failures))

    def test_docs_do_not_publish_machine_specific_paths_or_ssh_endpoints(self) -> None:
        failures: list[str] = []
        for document in self._documents():
            text = document.read_text(encoding="utf-8")
            for pattern in MACHINE_PATH_PATTERNS:
                if match := pattern.search(text):
                    failures.append(
                        f"{document.relative_to(ROOT)}: machine-specific path {match.group(0)}"
                    )
            if match := CONCRETE_SSH_ENDPOINT.search(text):
                failures.append(
                    f"{document.relative_to(ROOT)}: concrete SSH endpoint {match.group(0)}"
                )
        self.assertEqual(failures, [], "\n".join(failures))

    def test_current_protocol_is_unambiguous(self) -> None:
        technical = (ROOT / "docs-technical" / "README.md").read_text(encoding="utf-8")
        conceptual = (ROOT / "docs-conceptual" / "README.md").read_text(encoding="utf-8")
        history = (ROOT / "docs-technical" / "Development_History.md").read_text(encoding="utf-8")
        self.assertIn("active implementation", technical)
        self.assertIn("research decisions", conceptual)
        self.assertIn("archive/pre-paper-restructure-20260714", history)

    def test_shell_workflows_do_not_contain_placeholder_arguments(self) -> None:
        failures: list[str] = []
        for document in WORKFLOW_DOCUMENTS:
            in_shell_block = False
            for line_number, line in enumerate(document.read_text(encoding="utf-8").splitlines(), 1):
                if line.startswith("```bash") or line.startswith("```sh"):
                    in_shell_block = True
                    continue
                if in_shell_block and line.startswith("```"):
                    in_shell_block = False
                    continue
                if in_shell_block and (re.search(r"<[A-Za-z][^>]*>", line) or "..." in line):
                    failures.append(f"{document.relative_to(ROOT)}:{line_number}: {line}")
        self.assertEqual(failures, [], "\n".join(failures))


if __name__ == "__main__":
    unittest.main()
