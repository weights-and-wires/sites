from submission import format_query, build_inputs, add_eos

TASK = "Given a web search query, retrieve relevant passages that answer the query"


def test_format_query_exact():
    query = "how much protein should a female eat"
    got = format_query(TASK, query)
    want = "Instruct: " + TASK + "\nQuery: " + query
    assert got == want, f"expected {want!r}, got {got!r}"
    added = len(got) - len(query)
    assert added == 92, f"expected 92 added characters, got {added}"


def test_documents_have_no_prefix():
    docs = ["Protein needs vary.", "Instruct: this text only looks like a prefix"]
    qs, ds = build_inputs(TASK, ["a", "b"], docs)
    assert ds == docs, f"documents must be unchanged, got {ds}"
    assert qs == [format_query(TASK, "a"), format_query(TASK, "b")], f"every query needs the prefix, got {qs}"


def test_inputs_are_not_modified_and_outputs_are_new_lists():
    queries, docs = ["a"], ["d"]
    qs, ds = build_inputs(TASK, queries, docs)
    assert queries == ["a"] and docs == ["d"], f"inputs changed: {queries}, {docs}"
    assert ds is not docs, "doc_texts must be a new list"


def test_task_changes_only_the_query_side():
    _, d1 = build_inputs("task one", ["q"], ["doc"])
    _, d2 = build_inputs("task two", ["q"], ["doc"])
    q1, _ = build_inputs("task one", ["q"], ["doc"])
    q2, _ = build_inputs("task two", ["q"], ["doc"])
    assert d1 == d2, f"changing the task must not change the document side, got {d1} and {d2}"
    assert q1 != q2, f"changing the task must change the query side, got {q1} and {q2}"


def test_add_eos_appends_without_mutating():
    ids = [5, 6, 7]
    out = add_eos(ids, 2)
    assert out == [5, 6, 7, 2], f"expected [5, 6, 7, 2], got {out}"
    assert ids == [5, 6, 7], f"the input list must not change, got {ids}"
    assert out is not ids, "must return a new list"
