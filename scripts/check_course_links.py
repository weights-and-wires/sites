#!/usr/bin/env python3
"""Check that every course lesson URL on the papers site still exists.

Paper pages link to course lessons through content/papers/concepts.yaml and
each paper's course_links. A course can rename or renumber lessons, so this
script lists each course repository's lesson front matter on its default
branch and reports any URL whose slug no longer exists.

It reads the private course repositories through the GitHub API with a token
from COURSES_READ_TOKEN or GITHUB_TOKEN, or from `gh auth token`. Without a
token it prints a notice and exits 0, so CI without the secret still passes.

Usage:
    python3 scripts/check_course_links.py
"""

import json
import os
import pathlib
import re
import subprocess
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
# Lesson URL prefix -> (GitHub repository, lesson folder).
COURSES = {
    "https://ml.ifkash.dev/lessons/": ("weights-and-wires/ml", "content/lessons"),
    "https://rl.ifkash.dev/lessons/": ("weights-and-wires/rl", "content/lessons"),
    "https://speech.ifkash.dev/lessons/": ("weights-and-wires/speech", "content/lessons"),
    "https://gpu.ifkash.dev/c/": ("kashifulhaque/learn-gpu", "content/chapters"),
    "https://qwen.ifkash.dev/c/": ("kashifulhaque/learn-inference", "content/chapters"),
}
URL = re.compile(r"https://(?:ml|rl|speech|gpu|qwen)\.ifkash\.dev/(?:lessons|c)/[A-Za-z0-9_-]+")


def token():
    for name in ("COURSES_READ_TOKEN", "GITHUB_TOKEN"):
        if os.environ.get(name):
            return os.environ[name]
    try:
        return subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def api(path, tok):
    req = urllib.request.Request(
        "https://api.github.com/" + path,
        headers={"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def slugs(repo, folder, tok):
    """Lesson slugs are the markdown file names, which match each lesson's slug field."""
    entries = api(f"repos/{repo}/contents/{folder}", tok)
    return {e["name"][:-3] for e in entries if e["type"] == "file" and e["name"].endswith(".md")}


def main():
    tok = token()
    if not tok:
        print("check_course_links: no GitHub token; skipped")
        return
    urls = set()
    for f in [ROOT / "content/papers/concepts.yaml", *ROOT.glob("content/papers/*/index.md")]:
        urls |= set(URL.findall(f.read_text()))
    known = {}
    for prefix, (repo, folder) in COURSES.items():
        try:
            known[prefix] = slugs(repo, folder, tok)
        except Exception as exc:  # a token without access to one repo
            print(f"check_course_links: can't list {repo}: {exc}; skipped its URLs")
    broken = []
    for u in sorted(urls):
        for prefix, names in known.items():
            if u.startswith(prefix) and u[len(prefix):] not in names:
                broken.append(u)
    print(f"check_course_links: {len(urls)} course URLs, {len(broken)} broken")
    for u in broken:
        print("  " + u)
    sys.exit(1 if broken else 0)


if __name__ == "__main__":
    main()
