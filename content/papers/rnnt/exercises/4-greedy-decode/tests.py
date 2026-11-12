import numpy as np
from submission import greedy_decode

# The toy from the page. exp(f_t) and exp(g) after the start (0), an a (1) and a b (2).
F = np.array([[4, 4, 4], [6, 2, 1], [4, 4, 2]], dtype=float)
G = {0: [5, 4, 1], 1: [5, 1, 1], 2: [5, 1, 4]}


def page_joint(t, prefix):
    w = F[t] * np.array(G[prefix[-1] if prefix else 0], dtype=float)
    return w / w.sum()


def test_page_example_decodes_to_the_empty_sequence():
    got = greedy_decode(page_joint, 3)
    assert got == (), f"expected () (the null wins at all three frames), got {got}"


def test_returns_a_tuple_of_ints():
    got = greedy_decode(lambda t, p: np.array([0.1, 0.9]), 1, max_symbols=2)
    assert isinstance(got, tuple), f"expected a tuple, got {type(got).__name__}"
    assert all(isinstance(k, int) for k in got), f"expected ints, got {got}"


def test_prefix_changes_the_choice_within_a_frame():
    # At frame 0: after nothing emit a (1); after a emit b (2); after ab emit null.
    table = {(): [0.1, 0.8, 0.1], (1,): [0.1, 0.1, 0.8], (1, 2): [0.9, 0.05, 0.05]}
    got = greedy_decode(lambda t, p: np.array(table[p]), 1)
    assert got == (1, 2), f"expected (1, 2): a, then b given a, then null, got {got}"


def test_frames_are_visited_in_order_with_the_growing_prefix():
    calls = []

    def joint(t, prefix):
        calls.append((t, prefix))
        # Emit label 1 once at frame 1, otherwise null.
        if t == 1 and prefix == ():
            return np.array([0.2, 0.8])
        return np.array([0.9, 0.1])

    got = greedy_decode(joint, 3)
    want_calls = [(0, ()), (1, ()), (1, (1,)), (2, (1,))]
    assert got == (1,), f"expected (1,), got {got}"
    assert calls == want_calls, f"expected joint calls {want_calls}, got {calls}"


def test_cap_on_labels_per_frame():
    always_one = lambda t, p: np.array([0.1, 0.9])
    got = greedy_decode(always_one, 4, max_symbols=2)
    assert got == (1,) * 8, f"expected 2 labels in each of 4 frames (8 total), got {got}"
    got = greedy_decode(always_one, 2, max_symbols=1)
    assert got == (1, 1), f"expected one label per frame when max_symbols=1, got {got}"


def test_default_cap_is_three():
    always_two = lambda t, p: np.array([0.05, 0.15, 0.8])
    got = greedy_decode(always_two, 2)
    assert got == (2,) * 6, f"expected 3 labels per frame over 2 frames, got {got}"


def test_null_wins_a_tie_only_if_it_is_the_first_maximum():
    # np.argmax returns the first maximum, so a tie between null and a label picks the null.
    got = greedy_decode(lambda t, p: np.array([0.5, 0.5]), 2)
    assert got == (), f"expected () for a null-label tie (argmax takes index 0), got {got}"
