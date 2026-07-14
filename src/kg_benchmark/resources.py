from __future__ import annotations

import sys
from pathlib import Path


def artifact_path(relative_path: str | Path) -> Path:
    """Locate a repository artifact in a source checkout or an installed wheel."""
    relative = Path(relative_path)
    if relative.is_absolute():
        return relative
    module = Path(__file__).resolve()
    candidates = [Path.cwd() / relative]
    candidates.extend(parent / relative for parent in module.parents)
    candidates.append(Path(sys.prefix) / relative)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Cannot locate packaged artifact {relative}; searched: {searched}")
