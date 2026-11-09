# Write a paper page

This guide is the content contract for papers.ifkash.dev. Every page in
`content/papers/<id>/` follows it, and the build fails on most violations.

## Files

A paper folder holds the following files:

```text
content/papers/<id>/
  index.md                       # front matter plus the page body
  scripts/*.py                   # scripts that compute every number on the page
  exercises/<n>-<slug>/
    meta.json                    # title, goal, hints, runtime
    starter.py                   # what the reader starts from; fails the tests
    tests.py                     # plain test_ functions with assert messages
    solution.py                  # passes every test
```

## Front matter

The JSON schema is [`paper.schema.json`](paper.schema.json). The fields are as
follows:

| Field | Type | Notes |
|---|---|---|
| `id` | string | Must match the folder name. |
| `title`, `authors`, `year`, `venue` | | From the paper. |
| `arxiv` | string or `null` | The arXiv ID. Set `url` when there's none. |
| `url` | string | The paper's official URL. |
| `track` | string | One of the 11 tracks in `internal/content`. |
| `tier` | string | `core` or `extended`. |
| `est_hours` | number | Hours to read the page and do the exercises. |
| `status` | string | `stub` or `full`. A full page needs at least one exercise and one `self_check` item. |
| `summary` | string | The TL;DR: three sentences, plain text, no math. |
| `related_course` | string | A loose hint such as `ml: optimizers`. |
| `prereqs` | list | `{id, reason}` items. On a full page every `reason` is one sentence that says what this paper needs from that one. |
| `background` | list | Concept IDs from `content/papers/concepts.yaml`. |
| `course_links` | list | `{title, url}` course lessons that teach this paper in depth. |
| `self_check` | list | Three `{q, a, see}` items. `see` is the paper ID or concept ID that covers the question. `q` and `a` are markdown and can hold math. |

Don't change `prereqs` IDs or `background` IDs without a reason in the PR. The
prerequisite graph is reviewed as a whole.

## Body

The template renders the header, TL;DR, the "Before you read" panel, "What came
next", and "Course links". The body supplies everything else, in this order:

1. `## The problem`: what was broken before this paper, with a concrete failing
   example.
1. `## Key ideas`, with one `###` heading per idea. Each idea gives the
   intuition, then the math in the paper's notation, then a tiny numeric example
   that the reader can check by hand.
1. The line `<!-- exercises -->`. The exercises render here under
   "Implement it".
1. `## Pitfalls`: common misreadings and implementation bugs.
1. `## Further reading`: official code, a well-known blog post or lecture. This
   section renders after the course links.

Use `$...$` for inline math and `$$...$$` on its own lines for display math.
KaTeX renders it at build time, and bad TeX fails the build. Cite the paper's
section and equation numbers, for example "(Section 3.2.1, Equation 1)". When
your notation differs from the paper's, say so.

Every number in a numeric example must come from a script in `scripts/` that
prints it. Name the script near the example, for example "computed by
`scripts/attention_example.py`".

## Exercises

Each paper has two to four exercises.

`meta.json` has the following shape:

```json
{
  "title": "Causal mask",
  "goal": "Markdown that states the goal and the function signature.",
  "hints": ["Markdown for hint 1.", "Hint 2."],
  "runtime": "pyodide"
}
```

Set `runtime` to `torch` only when the exercise needs PyTorch. A PyTorch
exercise has no in-browser runner; the page offers a zip of the starter and
tests instead. Add `"lab": "COURSE_LAB_URL"` when a course lab covers the same
work.

The rules for the code are as follows:

- `tests.py` starts with `from submission import ...`. The runner saves the
  reader's code as `submission.py`.
- Each test is a `test_` function. Every `assert` has a message that shows the
  expected and actual values, for example
  `assert np.allclose(w, want), f"expected {want}, got {w}"`.
- `starter.py` must fail at least one test. Raise `NotImplementedError` in the
  bodies the reader writes, and keep signatures, docstrings, and imports.
- `solution.py` must pass every test.
- A `pyodide` exercise imports only NumPy and the standard library.
- Tests catch plausible wrong implementations, not just the shape of the output.
  For example, test that the softmax runs over the right axis and that a mask
  uses a large negative number rather than zero.
- Each run takes under 5 seconds on CPU (30 seconds for `torch`).

## Validate

Run these commands from the repository root before you open a PR:

```bash
go run ./cmd/build -offline -check
```

```bash
python3 scripts/check_exercises.py content/papers/PAPER_ID
```

Replace `PAPER_ID` with the folder name. Also run every script in
`content/papers/PAPER_ID/scripts/` and confirm its output matches the page.
