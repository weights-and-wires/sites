#!/usr/bin/env python3
"""Import the curated paper list as stub pages and reading paths.

The source is the research report (papers.md) that lists every paper as YAML
in section 2 and the reading paths in section 3. This script writes:

- content/papers/<id>/index.md for every paper that has no page yet, with
  status: stub. It never overwrites an existing page.
- content/paths/<id>.md for each reading path.

Usage:
    python3 scripts/import_papers.py PAPERS_MD [--related RELATED_YAML]

RELATED_YAML maps a paper ID to the course lessons that cover it, in the
shape {"related": {id: [{url, title}]}}.
"""

import argparse
import pathlib
import re
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent

PATH_IDS = {
    "3.1": "transformer-to-llm",
    "3.2": "flashattention",
    "3.3": "rl-to-reasoning",
    "3.4": "speech",
    "3.5": "serving",
}


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def quote(text):
    return '"' + str(text).replace("\\", "\\\\").replace('"', '\\"') + '"'


def stub(p, by_id, related):
    lines = ["---"]
    for key in ("id", "title", "authors"):
        lines.append(f"{key}: {quote(p[key])}")
    lines.append(f"year: {p['year']}")
    lines.append(f"venue: {quote(p['venue'])}")
    lines.append(f"arxiv: {quote(p['arxiv']) if p['arxiv'] else 'null'}")
    lines.append(f"url: {quote(p['url'])}")
    lines.append(f"track: {p['track']}")
    lines.append(f"tier: {p['tier']}")
    lines.append(f"est_hours: {p['est_hours']}")
    lines.append("status: stub")
    lines.append(f"summary: {quote(p['why'])}")
    lines.append(f"related_course: {quote(p['related_course'])}")
    if p["prereqs"]:
        lines.append("prereqs:")
        for r in p["prereqs"]:
            reason = by_id[r]["key_ideas"][0]
            lines.append(f"  - id: {r}")
            lines.append(f"    reason: {quote(reason)}")
    else:
        lines.append("prereqs: []")
    lines.append("background: [" + ", ".join(slug(b) for b in p["background"]) + "]")
    links = related.get(p["id"], [])
    if links:
        lines.append("course_links:")
        for link in links:
            lines.append(f"  - title: {quote(link['title'])}")
            lines.append(f"    url: {quote(link['url'])}")
    lines.append("---")
    lines.append("")
    lines.append("## Page in progress")
    lines.append("")
    lines.append(
        "This page is a placeholder. The full page adds worked examples and "
        "exercises that run in your browser. Until then, the prerequisite panel "
        "shows what to read first."
    )
    lines.append("")
    lines.append("## What the full page covers")
    lines.append("")
    for idea in p["key_ideas"]:
        lines.append(f"- {idea}")
    lines.append("")
    return "\n".join(lines)


def paths(md):
    section = md.split("## 3. Reading paths", 1)[1].split("## 4.", 1)[0]
    out = []
    for block in re.split(r"^### ", section, flags=re.M)[1:]:
        head, _, body = block.partition("\n")
        number, title = head.split(" ", 1)
        intro = body.strip().split("\n\n", 1)[0].strip()
        ids = re.findall(r"^\d+\. `([a-z0-9-]+)`", body, flags=re.M)
        out.append((PATH_IDS[number], title.strip(), intro, ids))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("papers_md")
    ap.add_argument("--related")
    args = ap.parse_args()

    md = pathlib.Path(args.papers_md).read_text()
    data = yaml.safe_load(md.split("```yaml\n", 1)[1].split("\n```", 1)[0])
    papers = data["papers"]
    by_id = {p["id"]: p for p in papers}
    related = {}
    if args.related:
        related = yaml.safe_load(pathlib.Path(args.related).read_text())["related"]

    made = 0
    for p in papers:
        page = ROOT / "content/papers" / p["id"] / "index.md"
        if page.exists():
            continue
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(stub(p, by_id, related))
        made += 1
    print(f"wrote {made} stub pages ({len(papers)} papers in the list)")

    pdir = ROOT / "content/paths"
    pdir.mkdir(parents=True, exist_ok=True)
    for pid, title, intro, ids in paths(md):
        missing = [i for i in ids if i not in by_id]
        if missing:
            sys.exit(f"path {pid}: unknown papers {missing}")
        assumes = intro.split(". ")[-1].rstrip(".") + "." if "Assumes" in intro else ""
        lines = ["---", f"id: {pid}", f"title: {quote(title)}"]
        if assumes:
            lines.append(f"assumes: {quote(assumes)}")
        lines.append("papers: [" + ", ".join(ids) + "]")
        lines += ["---", "", intro, ""]
        (pdir / f"{pid}.md").write_text("\n".join(lines))
    print(f"wrote {len(PATH_IDS)} reading paths")


if __name__ == "__main__":
    main()
