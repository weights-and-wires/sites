import numpy as np
from submission import mask_tokens

VOCAB, MASK, NSPECIAL = 1000, 3, 4


def make(n=100_000, seed=0):
    rng = np.random.default_rng(seed)
    ids = rng.integers(NSPECIAL, VOCAB, n)
    return ids, rng


def test_selection_rate_is_fifteen_percent():
    ids, rng = make()
    inputs, labels = mask_tokens(ids, VOCAB, MASK, rng)
    rate = float(np.mean(labels != -100))
    assert abs(rate - 0.15) < 0.01, f"expected about 15% of positions selected, got {rate:.4f}"


def test_labels_are_the_originals_at_selected_positions_only():
    ids, rng = make(5000, seed=1)
    inputs, labels = mask_tokens(ids, VOCAB, MASK, rng)
    sel = labels != -100
    assert np.array_equal(labels[sel], ids[sel]), "labels at selected positions must be the original ids"
    assert np.all(labels[~sel] == -100), "labels outside the selected positions must be -100"
    assert np.array_equal(inputs[~sel], ids[~sel]), "positions that are not selected must be unchanged"


def test_eighty_ten_ten_split():
    ids, rng = make()
    inputs, labels = mask_tokens(ids, VOCAB, MASK, rng)
    sel = labels != -100
    k = sel.sum()
    masked = float(np.sum(inputs[sel] == MASK) / k)
    same = float(np.sum(inputs[sel] == ids[sel]) / k)
    random_changed = float(np.sum((inputs[sel] != MASK) & (inputs[sel] != ids[sel])) / k)
    assert abs(masked - 0.8) < 0.02, f"expected about 80% of selected positions to be [MASK], got {masked:.3f}"
    # About 10% are kept; a random replacement equals the original only 1 time in 996.
    assert abs(same - 0.10) < 0.02, f"expected about 10% of selected positions unchanged, got {same:.3f}"
    assert abs(random_changed - 0.10) < 0.02, f"expected about 10% replaced by a random token, got {random_changed:.3f}"


def test_random_replacements_are_regular_tokens():
    ids, rng = make()
    inputs, labels = mask_tokens(ids, VOCAB, MASK, rng)
    changed = (inputs != ids) & (inputs != MASK)
    vals = inputs[changed]
    assert len(vals) > 100, f"expected many random replacements, got {len(vals)}"
    assert vals.min() >= NSPECIAL and vals.max() < VOCAB, f"random ids must be in [{NSPECIAL}, {VOCAB}), got range [{vals.min()}, {vals.max()}]"


def test_special_tokens_are_never_selected():
    rng = np.random.default_rng(2)
    ids = np.tile(np.array([0, 1, 2, 3, 50, 60, 70, 80]), 5000)
    inputs, labels = mask_tokens(ids, VOCAB, MASK, rng, mlm_prob=0.5)
    special = ids < NSPECIAL
    assert np.all(labels[special] == -100), "special tokens must not get a label"
    assert np.array_equal(inputs[special], ids[special]), "special tokens must not change"
    rate = float(np.mean(labels[~special] != -100))
    assert abs(rate - 0.5) < 0.03, f"expected about 50% of regular tokens selected with mlm_prob=0.5, got {rate:.3f}"


def test_input_is_not_modified():
    ids, rng = make(2000, seed=3)
    before = ids.copy()
    mask_tokens(ids, VOCAB, MASK, rng, mlm_prob=0.5)
    assert np.array_equal(ids, before), "ids was modified in place"


def test_same_seed_gives_same_output():
    ids, _ = make(1000, seed=4)
    a = mask_tokens(ids, VOCAB, MASK, np.random.default_rng(9))
    b = mask_tokens(ids, VOCAB, MASK, np.random.default_rng(9))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]), "the same seed must give the same corruption"
