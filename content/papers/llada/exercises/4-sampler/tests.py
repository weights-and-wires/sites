import numpy as np
from submission import generate

V = 5
M = 5
TARGET = np.array([2, 0, 3, 1, 2, 0])
BASE = np.array([0.9, 0.4, 0.9, 0.4, 0.9, 0.4])
L = len(TARGET)


def make_predict(log=None):
    """A deterministic toy model. Position i puts probability c on TARGET[i] and 1 - c on a wrong
    token, where c = min(0.95, BASE[i] + 0.15 * the number of unmasked neighbors)."""

    def predict(x):
        if log is not None:
            log.append(np.array(x).tolist())
        logits = np.full((L, V), np.log(1e-9))
        for i in range(L):
            nb = sum(1 for j in (i - 1, i + 1) if 0 <= j < L and x[j] != M)
            c = min(0.95, BASE[i] + 0.15 * nb)
            logits[i, TARGET[i]] = np.log(c)
            logits[i, (TARGET[i] + 1) % V] = np.log(1 - c)
        return logits

    return predict


def test_low_confidence_recovers_the_target_in_three_steps():
    got = generate(make_predict(), L, 3, M)
    assert got.tolist() == TARGET.tolist(), f"expected {TARGET.tolist()}, got {got.tolist()}"


def test_trace_of_the_inputs_the_model_sees():
    log = []
    generate(make_predict(log), L, 3, M)
    want = [[M] * 6, [2, M, 3, M, M, M], [2, 0, 3, M, 2, M]]
    assert log == want, f"the model must see {want} at the three steps, got {log}"


def test_unmasked_count_schedule_uses_integer_floor():
    log = []
    generate(make_predict(log), L, 4, M)
    counts = [sum(1 for v in row if v != M) for row in log]
    assert counts == [0, 1, 3, 4], f"6 * k // 4 gives 1, 3, 4, 6 unmasked after steps 1-4; the model saw {counts} before them"


def test_one_step_per_token_and_one_big_step():
    log = []
    generate(make_predict(log), L, L, M)
    counts = [sum(1 for v in row if v != M) for row in log]
    assert counts == [0, 1, 2, 3, 4, 5], f"expected one more unmasked token per step, got {counts}"
    log = []
    got = generate(make_predict(log), L, 1, M)
    assert len(log) == 1, f"one step means one model call, got {len(log)}"
    want = make_predict()(np.full(L, M)).argmax(axis=-1)
    assert got.tolist() == want.tolist(), f"one step must return the all-mask argmax {want.tolist()}, got {got.tolist()}"
    assert got.tolist() != TARGET.tolist(), "one-shot decoding commits the hard positions too early"


def test_output_has_no_masks_for_both_strategies():
    for remask in ("low_confidence", "random"):
        got = generate(make_predict(), L, 3, M, remask=remask, rng=np.random.default_rng(0))
        assert (got != M).all(), f"{remask}: the final sequence must have no mask, got {got.tolist()}"


def test_random_remasking_is_seeded_and_worse_on_this_model():
    a = generate(make_predict(), L, 3, M, "random", np.random.default_rng(4))
    b = generate(make_predict(), L, 3, M, "random", np.random.default_rng(4))
    assert np.array_equal(a, b), "the same seed must give the same sequence"
    accs = [(generate(make_predict(), L, 3, M, "random", np.random.default_rng(s)) == TARGET).mean() for s in range(300)]
    assert 0.6 < np.mean(accs) < 0.9, f"random remasking should average 0.6 to 0.9 token accuracy here, got {np.mean(accs):.3f}"
    assert max(accs) == 1.0 and min(accs) < 1.0, "some random runs succeed and some fail on this model"


def test_unknown_strategy_raises():
    try:
        generate(make_predict(), L, 3, M, remask="greedy")
    except ValueError:
        return
    raise AssertionError("an unknown remasking strategy must raise ValueError")
