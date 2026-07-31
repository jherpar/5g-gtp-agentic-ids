"""Guards the plan's risk #7 mitigation: label validation (Level 3 traffic
pattern checks in preprocessing/labeling.py) and the detection agents
(agents/rules.py) must never share code or import from each other, so label
quality is never validated using the same logic being evaluated as a
detector.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "agente_5g"


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_agents_rules_does_not_import_labeling():
    imports = _imported_modules(SRC / "agents" / "rules.py")
    assert not any("preprocessing.labeling" in m for m in imports)


def test_labeling_does_not_import_agents_rules():
    imports = _imported_modules(SRC / "preprocessing" / "labeling.py")
    assert not any("agents.rules" in m or m == "agents" for m in imports)


def test_thresholds_and_label_patterns_configs_are_separate_files():
    thresholds = SRC.parent.parent / "configs" / "thresholds.yaml"
    label_patterns = SRC.parent.parent / "configs" / "label_patterns.yaml"
    assert thresholds.exists()
    assert label_patterns.exists()
    assert thresholds != label_patterns
