"""The built `akeratos` wheel's contents and metadata.

The wheel holds the product package alone, its Requires-Python is the floor, no development tool
or forbidden runtime is a requirement of it or of any extra, its extras and entry points are the
expected ones, and its licence metadata agrees with the project's.
"""

import re
import zipfile
from email.parser import Parser

import pytest

# Checked against every Requires-Dist line, extras included: no extra may
# declare these directly. transformers reaches `eval` only transitively,
# through mlx-lm.
FORBIDDEN_RUNTIME = {
    "torch",
    "transformers",
    "pytest",
    "pytest-socket",
    "pytest-cov",
    "ruff",
    "pyright",
    "deptry",
    "import-linter",
    "vulture",
}
_NAME = re.compile(r"\s*([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)")


def requirement_name(requirement: str) -> str:
    """The PEP 503 normalised project name at the start of a requirement."""
    match = _NAME.match(requirement)
    assert match, requirement
    return re.sub(r"[-_.]+", "-", match.group(1)).lower()


EXTRAS = {"cpu", "cuda13", "eval", "ifeval", "wandb", "plot"}


def _metadata(whl):
    with zipfile.ZipFile(whl) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".dist-info/METADATA"))
        return Parser().parsestr(zf.read(name).decode()), zf.namelist()


def _entry_points(whl) -> dict[str, str]:
    with zipfile.ZipFile(whl) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".dist-info/entry_points.txt"))
        text = zf.read(name).decode()
    scripts, section = {}, None
    for line in text.splitlines():
        entry = line.strip()
        if entry.startswith("["):
            section = entry.strip("[]")
        elif "=" in entry and section == "console_scripts":
            key, value = (part.strip() for part in entry.split("=", 1))
            scripts[key] = value
    return scripts


def test_only_product_package(wheels) -> None:
    _, names = _metadata(wheels["akeratos"])
    tops = {n.split("/")[0] for n in names}
    assert tops == {"akeratos", "akeratos-0.1.1.dist-info"}, tops


def test_requires_python_is_floor(wheels) -> None:
    meta, _ = _metadata(wheels["akeratos"])
    assert meta["Requires-Python"] == ">=3.12"


@pytest.mark.parametrize(
    ("requirement", "name"),
    [
        ("torch~=2.14", "torch"),
        ("torch!=2.13", "torch"),
        ("torch (>=2.14)", "torch"),
        ("Torch>=2.14; extra == 'eval'", "torch"),
        ("mlx[cpu]>=0.32.4,<0.33; sys_platform == 'linux' and extra == 'cpu'", "mlx"),
        ("pytest_socket>=0.7", "pytest-socket"),
        ("import.linter", "import-linter"),
        ("transformers", "transformers"),
    ],
)
def test_requirement_name(requirement: str, name: str) -> None:
    assert requirement_name(requirement) == name


def test_no_dev_tool_in_requires_dist(wheels) -> None:
    meta, _ = _metadata(wheels["akeratos"])
    requires = meta.get_all("Requires-Dist")
    assert requires
    bad = [r for r in requires if requirement_name(r) in FORBIDDEN_RUNTIME]
    assert not bad, bad


def test_provides_extra(wheels) -> None:
    meta, _ = _metadata(wheels["akeratos"])
    extras = meta.get_all("Provides-Extra")
    assert extras is not None
    assert set(extras) == EXTRAS


def test_entry_points_are_cli_names(wheels) -> None:
    scripts = _entry_points(wheels["akeratos"])
    assert scripts == {"bake": "akeratos.cli:main", "akeratos": "akeratos.cli:main"}


def test_licence_agrees(root, pyproject, wheels) -> None:
    licence = pyproject["project"]["license"]
    assert licence == "Apache-2.0"
    assert pyproject["project"]["license-files"] == ["LICENSE"]
    text = (root / "LICENSE").read_text()
    assert "Apache License" in text
    assert "Version 2.0" in text
    meta, names = _metadata(wheels["akeratos"])
    assert meta["License-Expression"] == licence
    assert "akeratos-0.1.1.dist-info/licenses/LICENSE" in names
