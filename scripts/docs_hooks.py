"""MkDocs hooks for the documentation site (mkdocs.yml, `hooks:`).

The Markdown of this repository is written for GitHub, and the site is built from the same files. Three differences are bridged here, at
build time, so that each file keeps ONE source that renders on both:

1. Mathematics. GitHub consumes the backslash escapes of punctuation before it reads the formula, so inline formulas are written with
   doubled backslashes (`$a\\,b$`), displayed formulas are fenced as ```math, and `$`...`$` is accepted. MathJax wants the plain LaTeX:
   inline `\\x` becomes `\\x` with one backslash, ```math fences become `$$ ... $$`, `$`...`$` becomes `$...$`.
2. Lists and tables. GitHub starts a list or a table right after a paragraph line and accepts 2 or 3 spaces for a nested list;
   Python-Markdown wants a blank line before and 4 spaces. Both are added.
3. Figures. The figures live in results/toy_final_report/ (they are produced there by scripts/make_report.py); they are added to the
   site under assets/figures/ instead of being copied into docs/.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "results" / "toy_final_report"
FIGURES_URI = "assets/figures"

_TOKEN = re.compile(r"\$`(.+?)`\$|(`+)(?:(?!\2).)+?\2")
_INLINE_MATH = re.compile(r"(?<![\\$])\$(?![\s$])(.+?)(?<![\s\\$])\$(?![\d$])")
_FENCE = re.compile(r"^(\s*)(`{3,}|~{3,})\s*([\w-]*)")
_ITEM = re.compile(r"^(\s*)([-*+]|\d+\.)\s")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")


def on_files(files, config):
    """Serve the figures of the toy report next to the pages."""
    from mkdocs.structure.files import File
    for path in sorted(FIGURES.glob("fig*.png")):
        files.append(File.generated(config, src_uri=f"{FIGURES_URI}/{path.name}", abs_src_path=str(path)))
    return files


def _inline(line: str) -> str:
    """Plain LaTeX in the inline formulas of one line; code spans are left alone."""
    out, last = [], 0
    for tok in _TOKEN.finditer(line):                  # left to right: a code span or a `$`...`$` formula, whichever starts first
        out.append(_math(line[last:tok.start()]))
        out.append(f"${tok.group(1)}$" if tok.group(1) is not None else tok.group(0))
        last = tok.end()
    out.append(_math(line[last:]))
    return "".join(out)


def _math(text: str) -> str:
    return _INLINE_MATH.sub(lambda m: "$" + m.group(1).replace("\\\\", "\\") + "$", text)


def on_page_markdown(markdown, page, config, files):
    out, fence, math = [], None, None
    for line in markdown.split("\n"):
        m = _FENCE.match(line)
        if fence is None:
            if m:
                fence = (m.group(2)[0], len(m.group(2)))
                if m.group(3) == "math":                         # ```math  ->  $$ ... $$
                    math = []
                    continue
                out.append(line)
                continue
            if line.strip() == "<br>":                           # spacing hack for GitHub
                continue
            line = _inline(line)
            item = _ITEM.match(line)
            prev = out[-1] if out else ""
            if item and 1 <= len(item.group(1)) <= 3:
                line = "    " + line.lstrip()                    # nested list: 4 spaces
            elif item and prev.strip() and not _ITEM.match(prev) and not prev.startswith((" ", "\t", ">")) and not _TABLE_ROW.match(prev):
                out.append("")                                   # a list right after a paragraph
            elif _TABLE_ROW.match(line) and prev.strip() and not _TABLE_ROW.match(prev):
                out.append("")                                   # a table right after a paragraph
            out.append(line)
        else:
            if m and m.group(3) == "" and m.group(2)[0] == fence[0] and len(m.group(2)) >= fence[1]:
                if math is not None:
                    out += ["", "$$", *math, "$$", ""]
                    math = None
                else:
                    out.append(line)
                fence = None
            elif math is not None:
                math.append(line)
            else:
                out.append(line)
    text = "\n".join(out)
    return text.replace("results/toy_final_report/fig", f"{FIGURES_URI}/fig")
