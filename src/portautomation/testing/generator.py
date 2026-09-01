"""Automatically generate smoke tests for package modules."""

from __future__ import annotations

import inspect
from pathlib import Path

from portautomation.config import PROJECT_ROOT

GENERATED_TEST_FILE = PROJECT_ROOT / "tests" / "test_auto_generated.py"
PACKAGE_DIR = PROJECT_ROOT / "src" / "portautomation"
SKIP_MODULES = {"webapp", "testing", "__pycache__"}


def _discover_modules() -> list[str]:
    modules: list[str] = []
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        if path.name.startswith("_"):
            continue
        stem = path.stem
        if stem in SKIP_MODULES:
            continue
        modules.append(f"portautomation.{stem}")
    return modules


def _public_functions(module_name: str) -> list[str]:
    import importlib

    module = importlib.import_module(module_name)
    names: list[str] = []
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if name.startswith("_"):
            continue
        if getattr(obj, "__module__", "").startswith("portautomation"):
            names.append(name)
    return sorted(names)


def _render_test_file(modules: list[str], functions_by_module: dict[str, list[str]]) -> str:
    lines = [
        '"""Auto-generated smoke tests. Regenerate with: python -m portautomation.testing.generator"""',
        "",
        "import importlib",
        "",
        "import pytest",
        "",
        "MODULES = " + repr(modules),
        "",
    ]

    for module_name in modules:
        safe = module_name.replace(".", "_")
        lines.extend(
            [
                f"def test_import_{safe}():",
                f"    module = importlib.import_module({module_name!r})",
                "    assert module is not None",
                "",
            ]
        )

    lines.extend(
        [
            "@pytest.mark.parametrize('module_name', MODULES)",
            "def test_module_has_docstring(module_name):",
            "    module = importlib.import_module(module_name)",
            "    assert module.__doc__ is not None",
            "",
        ]
    )

    for module_name, functions in functions_by_module.items():
        if not functions:
            continue
        safe = module_name.replace(".", "_")
        lines.extend(
            [
                f"def test_public_functions_exist_{safe}():",
                f"    module = importlib.import_module({module_name!r})",
                "    expected = " + repr(functions),
                "    for name in expected:",
                "        assert hasattr(module, name), f'Missing function: {name}'",
                "",
            ]
        )

    lines.extend(
        [
            "def test_config_paths_exist():",
            "    from portautomation import config",
            "    assert config.PROJECT_ROOT.exists()",
            "    assert config.NUM_CLASSES == 9",
            "    assert config.IMAGE_SIZE == (224, 224)",
            "",
            "def test_class_names_count_matches_num_classes():",
            "    from portautomation import config",
            "    assert len(config.CLASS_NAMES) == config.NUM_CLASSES",
            "",
        ]
    )
    return "\n".join(lines)


def generate_tests(output_path: Path | None = None) -> Path:
    """Generate smoke tests for discovered package modules."""
    output_path = output_path or GENERATED_TEST_FILE
    modules = _discover_modules()
    functions_by_module = {module: _public_functions(module) for module in modules}
    content = _render_test_file(modules, functions_by_module)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path


def main() -> None:
    path = generate_tests()
    print(f"Generated tests at {path}")


if __name__ == "__main__":
    main()
