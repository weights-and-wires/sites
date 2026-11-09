import numpy as np
from submission import repeat_rates

# Four tokens. Pairs: (0,1) same first choice; (1,2) swapped, so the sets overlap
# but the first choices differ; (2,3) disjoint.
IDX = np.array([[3, 5], [3, 1], [1, 3], [6, 7]])


def test_hand_example():
    first, either = repeat_rates(IDX)
    assert abs(first - 1 / 3) < 1e-9, f"expected first = 1/3, got {first}"
    assert abs(either - 2 / 3) < 1e-9, f"expected either = 2/3, got {either}"


def test_swapped_pair_counts_for_either_only():
    first, either = repeat_rates(np.array([[3, 5], [5, 3]]))
    assert first == 0.0, f"first choices 3 and 5 differ, expected first = 0, got {first}"
    assert either == 1.0, f"the sets are equal, expected either = 1, got {either}"


def test_full_overlap_is_one_not_two():
    _, either = repeat_rates(np.array([[3, 5], [3, 5], [3, 5]]))
    assert either == 1.0, f"a rate cannot exceed 1; expected 1.0, got {either}"


def test_second_slot_match_counts():
    # The first choice of token 1 equals the second choice of token 0.
    _, either = repeat_rates(np.array([[0, 4], [4, 7]]))
    assert either == 1.0, f"expected 1.0 for a shared expert in different slots, got {either}"


def test_uses_adjacent_tokens_only():
    # Tokens 0 and 2 agree, but they are not neighbors.
    first, either = repeat_rates(np.array([[1, 2], [3, 4], [1, 2]]))
    assert first == 0.0 and either == 0.0, f"expected (0, 0) for alternating routes, got ({first}, {either})"


def test_random_routing_matches_the_baselines():
    rng = np.random.default_rng(0)
    idx = np.array([rng.permutation(8)[:2] for _ in range(20000)])
    first, either = repeat_rates(idx)
    assert abs(first - 0.125) < 0.01, f"expected about 0.125, got {first}"
    assert abs(either - 0.4643) < 0.015, f"expected about 0.4643, got {either}"
