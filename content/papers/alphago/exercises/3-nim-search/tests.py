import numpy as np
from submission import mcts


def exact_value(n):
    return -1.0 if n % 4 == 0 else 1.0


def flat_value(n):
    return 0.0


def uniform(n):
    return np.ones(3) / 3


def favor_take_one(n):
    return np.array([0.9, 0.05, 0.05])


def test_visit_counts_add_up():
    c = np.asarray(mcts(7, uniform, exact_value, 60))
    assert c.shape == (3,), f"expected shape (3,), got {c.shape}"
    assert c.sum() == 60, f"expected 60 root visits, got {c.sum()}"


def test_exact_value_finds_the_winning_move():
    for n, want in ((5, 0), (6, 1), (7, 2)):
        c = np.asarray(mcts(n, uniform, exact_value, 100))
        assert int(np.argmax(c)) == want, f"with {n} stones the winning move has index {want}, got visits {c}"


def test_lost_position_has_no_winning_move():
    # At 4 stones every move leaves a winning position for the opponent.
    c = np.asarray(mcts(4, uniform, exact_value, 100))
    assert c.sum() == 100, f"expected 100 visits, got {c.sum()}"
    assert c.max() < 70, f"no move is clearly best at 4 stones, but one got {c.max()} of 100 visits: {c}"


def test_illegal_moves_get_no_visits():
    c = np.asarray(mcts(2, uniform, exact_value, 40))
    assert c[2] == 0, f"taking 3 stones from 2 is illegal, got visits {c}"
    assert c[1] > c[0], f"taking 2 stones wins at once, expected index 1 to lead, got {c}"
    c = np.asarray(mcts(1, uniform, exact_value, 10))
    assert np.allclose(c, [10, 0, 0]), f"with one stone only index 0 is legal, expected [10, 0, 0], got {c}"


def test_value_beats_a_wrong_prior_given_enough_simulations():
    c = np.asarray(mcts(7, favor_take_one, exact_value, 1000))
    assert int(np.argmax(c)) == 2, f"1000 simulations with exact values should overrule the prior, got visits {c}"


def test_prior_dominates_a_short_search():
    c = np.asarray(mcts(7, favor_take_one, exact_value, 10))
    assert int(np.argmax(c)) == 0, f"after 10 simulations the 0.9 prior should still lead, got visits {c}"


def test_search_without_a_value_network_still_finds_wins():
    # Only the terminal loss is known. The search must reach the end of the game
    # and back the result up with the right signs.
    for n, want in ((5, 0), (6, 1), (7, 2)):
        c = np.asarray(mcts(n, uniform, flat_value, 400))
        assert int(np.argmax(c)) == want, f"with {n} stones and no value network, expected index {want}, got visits {c}"
        assert c[want] > 0.6 * c.sum(), f"expected more than 60% of visits on index {want}, got {c}"


def test_wrong_sign_would_pick_a_losing_move():
    # With the exact value network, a search that forgets to flip signs prefers
    # moves that hand the opponent a winning position. The right search does not.
    c = np.asarray(mcts(6, uniform, exact_value, 200))
    assert c[1] > c[0] + c[2], f"expected index 1 (leaving 4 stones) to get most visits, got {c}"


def test_network_value_is_for_the_player_to_move_at_the_leaf():
    # The network says the mover at 6 stones wins (+1) and the mover at 4 or 5 stones
    # loses (-1). So taking 1 stone from 7 (leaving 6) is bad, and taking 2 or 3 is good.
    def value(n):
        return 1.0 if n == 6 else (-1.0 if n in (4, 5) else 0.0)

    c = np.asarray(mcts(7, uniform, value, 30, c_puct=1.0))
    assert c[0] <= 5, f"leaving a winning position to the opponent is bad, expected at most 5 visits on index 0, got {c}"
    assert c[1] + c[2] >= 25, f"expected at least 25 visits on indices 1 and 2, got {c}"
