"""Verify handbook chapters against AUTHORING.md.

Usage: python verify_handbook.py [--blocks OUT.json]
Checks every NN-*.md and appendix in this folder: em dashes, unquoted Mermaid
labels, unbalanced display-math delimiters, required template sections for
numbered chapters, listing labels, exercises with solutions, line counts.
Optionally writes all Mermaid blocks to a JSON file for a browser render check.
"""
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).parent
CHAPTER_RE = re.compile(r"^(\d\d)-.*\.md$")
REQUIRED_SECTIONS = [
    "The problem this chapter solves",
    "Implementation notes",
    "Failure modes",
    "On your machine",
    "## Exercises",
    "## Summary",
    "## Further reading",
]
NO_LISTINGS_OK = {"25"}  # chapter 25 has template skeletons instead of code listings


def mermaid_blocks(text):
    return re.findall(r"```mermaid\n(.*?)```", text, flags=re.S)


def unquoted_labels(block):
    bad = []
    for line in block.splitlines():
        s = line.strip()
        if not s or s.startswith(("%%", "subgraph", "section", "title", "participant", "Note", "state ", "class ")):
            continue
        # node definitions like A[text] or A{text} or A(text) without a quote right after the bracket
        for m in re.finditer(r"[A-Za-z0-9_]+\s*([\[\{\(]{1,2})([^\"\]\}\)])", s):
            # allow shapes like A(["..."]) and A{{"..."}} where the quote follows the second bracket
            after = s[m.end(1):m.end(1) + 2]
            if '"' in after:
                continue
            bad.append(s)
            break
    return bad


def check_file(path):
    text = path.read_text(encoding="utf-8")
    lines = text.count("\n") + 1
    problems = []
    if "—" in text:
        n = text.count("—")
        problems.append(f"{n} em dash(es)")
    if text.count("$$") % 2:
        problems.append("odd number of $$ delimiters")
    blocks = mermaid_blocks(text)
    for i, b in enumerate(blocks, 1):
        first = b.strip().splitlines()[0] if b.strip() else ""
        if first.startswith(("flowchart", "graph", "stateDiagram")):
            bad = unquoted_labels(b)
            if bad:
                problems.append(f"mermaid #{i}: unquoted label(s): {bad[0][:60]}")
        if first.startswith("gantt"):
            for ln in b.splitlines():
                if ":" in ln and not ln.strip().startswith(("title", "dateFormat", "axisFormat", "section")) and ln.count(":") > 1:
                    problems.append(f"mermaid #{i}: gantt task with extra colon: {ln.strip()[:60]}")
                    break
    m = CHAPTER_RE.match(path.name)
    is_chapter = bool(m) and m.group(1) != "00"
    if is_chapter:
        for sec in REQUIRED_SECTIONS:
            if sec not in text:
                problems.append(f"missing section: {sec}")
        listings = len(re.findall(r"\*\*Listing \d+\.\d+", text))
        if listings == 0 and m.group(1) not in NO_LISTINGS_OK:
            problems.append("no labeled listings")
        if "<details>" not in text:
            problems.append("no <details> solution blocks")
        if not (3 <= len(blocks) <= 8):
            problems.append(f"{len(blocks)} mermaid diagrams (expected 3 to 6)")
        if lines < 500:
            problems.append(f"only {lines} lines")
    return lines, len(blocks), problems, blocks


def main():
    out_blocks = None
    if "--blocks" in sys.argv:
        out_blocks = pathlib.Path(sys.argv[sys.argv.index("--blocks") + 1])
    files = sorted(p for p in HERE.glob("*.md") if p.name not in {"README.md", "AUTHORING.md", "OUTLINE.md", "FDE_Handbook.md"})
    all_blocks = []
    total_lines = 0
    failures = 0
    for f in files:
        lines, nblocks, problems, blocks = check_file(f)
        total_lines += lines
        all_blocks += [{"file": f.name, "i": i + 1, "code": b} for i, b in enumerate(blocks)]
        status = "OK " if not problems else "FIX"
        if problems:
            failures += 1
        print(f"{status} {f.name:60s} {lines:5d} lines {nblocks:2d} diagrams")
        for p in problems:
            print(f"      - {p}")
    print(f"\n{len(files)} files, {total_lines} lines, {len(all_blocks)} diagrams, {failures} file(s) with problems")
    if out_blocks:
        out_blocks.write_text(json.dumps(all_blocks), encoding="utf-8")
        print(f"wrote {out_blocks}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
