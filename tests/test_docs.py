"""The documentation: the hook that turns the GitHub-flavoured Markdown into MathJax input (scripts/docs_hooks.py) and the links of
README.md and docs/index.md. Needs neither mkdocs nor a browser; `mkdocs build --strict` (docs.yml) checks the rest."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import docs_hooks                                                    # noqa: E402

SITE = "https://josh012006.github.io/CIB-IRL/"


def convert(md: str) -> str:
    return docs_hooks.on_page_markdown(md, None, None, None)


def slug(heading: str) -> str:
    """The anchor GitHub gives to a heading."""
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


def headings(text: str) -> set:
    out, fence = set(), False
    for line in text.split("\n"):
        if line.startswith("```"):
            fence = not fence
        m = None if fence else re.match(r"^#{1,6} (.*)$", line)
        if m:
            out.add(slug(m.group(1)))
    return out


def test_inline_formulas_lose_the_github_escapes():
    assert convert(r"a $x\\,y$ and $\\{1,\dots,K\\}$ and $\\Big[a\\Big]$") == r"a $x\,y$ and $\{1,\dots,K\}$ and $\Big[a\Big]$"


def test_backtick_formulas_and_code_spans():
    assert convert(r"$`\mathrm{softmax}(R)`$") == r"$\mathrm{softmax}(R)$"
    assert convert(r"`$a\\,b$` stays code") == r"`$a\\,b$` stays code"
    assert convert("it costs $5 and $6") == "it costs $5 and $6"


def test_math_fences_become_displayed_formulas_and_code_fences_are_untouched():
    out = convert("<br>\n\n```math\n\\max_\\phi x\n```\n\n<br>\n\n```\ncode $a\\\\,b$\n```")
    assert "$$\n\\max_\\phi x\n$$" in out and "<br>" not in out and "```math" not in out
    assert "code $a\\\\,b$" in out


def test_figures_are_served_from_the_assets_folder():
    assert convert("![f](results/toy_final_report/fig1_task.png)") == "![f](assets/figures/fig1_task.png)"


def test_anchors_of_the_documentation_page_resolve():
    text = (ROOT / "docs" / "index.md").read_text(encoding="utf-8")
    heads = headings(text)
    missing = sorted({a for a in re.findall(r"\]\(#([^)]+)\)", text) if a not in heads})
    assert not missing, missing


def test_links_of_the_readme_resolve():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    docs = headings((ROOT / "docs" / "index.md").read_text(encoding="utf-8"))
    local = headings(readme)
    assert not sorted({a for a in re.findall(r"\]\(#([^)]+)\)", readme) if a not in local})
    assert not sorted({a for a in re.findall(re.escape(SITE) + r"#([^)\s]+)", readme) if a not in docs})


def test_the_readme_keeps_the_section_numbers_of_the_documentation():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    docs = (ROOT / "docs" / "index.md").read_text(encoding="utf-8")
    pattern = re.compile(r"^## (\d+)\. (.*)$", re.M)
    in_docs = dict(pattern.findall(docs))
    for number, title in pattern.findall(readme):
        assert in_docs.get(number) == title, (number, title)


def test_lists_and_tables_after_a_paragraph_get_the_blank_line_that_python_markdown_needs():
    out = convert("Reading the terms.\n- one\n- two\n\ntext\n| a | b |\n|---|---|\n| 1 | 2 |")
    assert out == "Reading the terms.\n\n- one\n- two\n\ntext\n\n| a | b |\n|---|---|\n| 1 | 2 |"


def test_nested_lists_are_indented_by_four_spaces_and_continuations_are_left_alone():
    out = convert("1. step\n   - sub a\n   - sub b\n2. next\n\n- [ ] item\n  continued text\n- [x] other")
    assert out == "1. step\n    - sub a\n    - sub b\n2. next\n\n- [ ] item\n    continued text\n- [x] other".replace("    continued", "  continued")
