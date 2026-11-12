import numpy as np
from submission import K, search, self_play_game, visits_to_policy


def exact_value(n):
    return -1.0 if n % 4 == 0 else 1.0


def uniform(n):
    return np.ones(K) / K


def play(n_start=7, sims=60, seed=0, tau_moves=2, prior=uniform):
    return self_play_game(n_start, prior, exact_value, sims, np.random.default_rng(seed), tau_moves)


def test_game_follows_the_rules_of_nim():
    for seed in range(5):
        game = play(10, seed=seed)
        states = [g[0] for g in game]
        assert states[0] == 10, f"seed {seed}: expected the first state to be 10, got {states}"
        steps = np.diff(states + [0])
        assert np.all((steps <= -1) & (steps >= -3)), f"seed {seed}: expected every move to take 1 to 3 stones, got states {states}"
        assert 1 <= states[-1] <= 3, f"seed {seed}: the last mover must take all remaining stones (1 to 3), got states {states}"


def test_z_alternates_and_the_last_mover_wins():
    for seed in range(5):
        game = play(10, seed=seed)
        zs = [g[2] for g in game]
        assert zs[-1] == 1.0, f"seed {seed}: the player who takes the last stone wins, expected last z = +1, got {zs}"
        want = [1.0 if (len(zs) - 1 - i) % 2 == 0 else -1.0 for i in range(len(zs))]
        assert zs == want, f"seed {seed}: expected z to alternate back from the last move, {want}, got {zs}"


def test_z_is_from_the_view_of_the_player_to_move():
    # With greedy play and the exact value function, 7 stones is a win for the first player.
    game = play(7, sims=60, tau_moves=0)
    assert game[0][0] == 7 and game[0][2] == 1.0, f"expected 7 stones to be a win for the mover (z = +1), got {game[0]}"
    game = play(8, sims=60, tau_moves=0)
    assert game[0][0] == 8 and game[0][2] == -1.0, f"expected 8 stones to be a loss for the mover (z = -1), got {game[0]}"


def test_pi_is_the_tau_one_visit_share_from_search():
    game = play(7, sims=50, seed=3, tau_moves=3)
    for n, pi, z in game:
        want = visits_to_policy(search(n, uniform, exact_value, 50), 1.0)
        assert np.allclose(pi, want, atol=1e-9), f"state {n}: expected pi = visit share {want}, got {pi}"
        assert abs(np.sum(pi) - 1.0) < 1e-9, f"state {n}: expected pi to sum to 1, got {np.sum(pi)}"


def test_pi_is_not_one_hot_on_greedy_moves():
    game = play(7, sims=50, tau_moves=0)
    pi = game[0][1]
    assert 0.0 < pi.max() < 0.99, f"expected the stored target to stay the tau = 1 visit share, got {pi}"


def test_pi_is_zero_on_illegal_moves():
    game = play(5, sims=40, seed=1, tau_moves=5)
    for n, pi, z in game:
        if n < 3:
            assert pi[2] == 0.0, f"state {n}: taking 3 stones is illegal, expected pi[2] = 0, got {pi}"
        if n < 2:
            assert np.allclose(pi, [1, 0, 0]), f"state {n}: expected pi = [1, 0, 0], got {pi}"


def test_greedy_moves_take_the_winning_move():
    for seed in range(10):
        game = play(7, sims=50, seed=seed, tau_moves=0)
        assert game[1][0] == 4, f"seed {seed}: expected the greedy move from 7 to leave 4 stones, got states {[g[0] for g in game]}"


def test_first_moves_are_sampled_from_pi():
    seconds = set()
    for seed in range(40):
        seconds.add(play(7, sims=10, seed=seed, tau_moves=1)[1][0])
    assert len(seconds) > 1, f"expected sampling to vary the first move across seeds, always got second state {seconds}"
    only = set(play(7, sims=10, seed=s, tau_moves=0)[1][0] for s in range(10))
    assert len(only) == 1, f"expected greedy play to be repeatable, got second states {only}"
