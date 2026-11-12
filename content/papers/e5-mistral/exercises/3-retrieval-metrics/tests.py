import math
from submission import recall_at_k, ndcg_at_k

RANKED = ["d3", "d1", "d7", "d2"]
REL = {"d1", "d2"}


def test_recall_hand_values():
    for k, want in [(1, 0.0), (2, 0.5), (3, 0.5), (4, 1.0)]:
        got = recall_at_k(RANKED, REL, k)
        assert math.isclose(got, want), f"recall@{k}: expected {want}, got {got}"


def test_recall_divides_by_relevant_count():
    got = recall_at_k(["a", "b"], {"a", "b", "c", "d", "e"}, 2)
    assert math.isclose(got, 0.4), f"expected 2/5 = 0.4, got {got}"


def test_ndcg_hand_values():
    d3 = 1 / math.log2(3)
    idcg = 1 + d3
    for k, want in [(1, 0.0), (2, d3 / idcg), (3, d3 / idcg), (4, (d3 + 1 / math.log2(5)) / idcg)]:
        got = ndcg_at_k(RANKED, REL, k)
        assert math.isclose(got, want, rel_tol=1e-9), f"ndcg@{k}: expected {want:.4f}, got {got}"


def test_ndcg_perfect_ranking_is_one_even_when_k_is_below_relevant_count():
    got = ndcg_at_k(["a", "b", "x"], {"a", "b", "c", "d", "e"}, 2)
    assert math.isclose(got, 1.0), f"the ideal top 2 is two relevant documents, so expected 1.0, got {got}"


def test_ndcg_discount_starts_at_log2_of_two():
    got = ndcg_at_k(["a", "x"], {"a"}, 2)
    assert math.isclose(got, 1.0), f"a relevant document at rank 1 with one relevant overall must give 1.0, got {got}"
    got2 = ndcg_at_k(["x", "a"], {"a"}, 2)
    assert math.isclose(got2, 1 / math.log2(3)), f"rank 2 is discounted by 1/log2(3) = 0.6309, got {got2}"


def test_documents_below_k_do_not_count():
    assert math.isclose(recall_at_k(["x", "y", "a"], {"a"}, 2), 0.0), "a document at rank 3 must not count for k=2"
    assert math.isclose(ndcg_at_k(["x", "y", "a"], {"a"}, 2), 0.0), "a document at rank 3 must not count for k=2"


def test_ranking_shorter_than_k_and_empty_relevant():
    got = ndcg_at_k(["a"], {"a", "b"}, 10)
    want = 1.0 / (1 + 1 / math.log2(3))
    assert math.isclose(got, want), f"expected {want:.4f}, got {got}"
    assert recall_at_k(["a"], set(), 3) == 0.0 and ndcg_at_k(["a"], set(), 3) == 0.0, "no relevant documents must give 0.0"


def test_order_matters_for_ndcg_not_for_recall():
    a, b = ["d1", "d2", "x"], ["x", "d1", "d2"]
    assert math.isclose(recall_at_k(a, REL, 3), recall_at_k(b, REL, 3)), "recall ignores order inside the top k"
    assert ndcg_at_k(a, REL, 3) > ndcg_at_k(b, REL, 3), f"nDCG must reward the earlier ranks; got {ndcg_at_k(a, REL, 3)} and {ndcg_at_k(b, REL, 3)}"
