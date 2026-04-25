"""Verify that requirements.txt stays in sync with pyproject.toml.

pyproject.toml is the single source of truth for dependency pins; the
requirements.txt mirror exists for tools that can only read that format
(Dockerfile, vast.ai bootstrap). This test catches silent drift between
the two lists.
"""

from __future__ import annotations

import re
from pathlib import Path

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
REQUIREMENTS = ROOT / "requirements.txt"

# Package name is normalised per PEP 503: lowercase, '-' and '_' treated equal.
_NAME_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)")


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _parse_requirement_line(line: str) -> str | None:
    line = line.split("#", 1)[0].strip()
    if not line or line.startswith("-"):
        return None
    m = _NAME_RE.match(line)
    return _normalize(m.group(1)) if m else None


def _load_requirements() -> set[str]:
    names: set[str] = set()
    for raw in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        name = _parse_requirement_line(raw)
        if name:
            names.add(name)
    return names


def _load_pyproject() -> set[str]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    deps: list[str] = list(data["project"].get("dependencies", []))
    for extra_deps in data["project"].get("optional-dependencies", {}).values():
        deps.extend(extra_deps)
    names: set[str] = set()
    for spec in deps:
        m = _NAME_RE.match(spec)
        if m:
            names.add(_normalize(m.group(1)))
    return names


def test_requirements_and_pyproject_list_same_packages() -> None:
    req = _load_requirements()
    pyp = _load_pyproject()

    missing_in_req = pyp - req
    extra_in_req = req - pyp

    msg_parts = []
    if missing_in_req:
        msg_parts.append(f"in pyproject but not requirements.txt: {sorted(missing_in_req)}")
    if extra_in_req:
        msg_parts.append(f"in requirements.txt but not pyproject: {sorted(extra_in_req)}")
    assert not msg_parts, "Dependency drift — " + "; ".join(msg_parts)
