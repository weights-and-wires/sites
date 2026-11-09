import numpy as np
from submission import nar_input, ada_layer_norm

K, V, d = 3, 4, 2
W = np.array([[[10 * (j + 1) + v, v] for v in range(V)] for j in range(K)], dtype=float)
TEXT = np.array([[0.5, -0.5]])
PROMPT = np.array([[1, 0, 2]])
TARGET = np.array([[3, 1, 0], [2, 0, 1]])


def test_stage_three_values():
    # Equations 4 and 5: sum codebooks 1 and 2 of the target; all three of the prompt.
    got = nar_input(TEXT, PROMPT, TARGET, W, 3)
    want = np.array([[0.5, -0.5], [63, 3], [34, 4], [32, 2]], dtype=float)
    assert got.shape == (4, 2), f"expected shape (4, 2), got {got.shape}"
    assert np.allclose(got, want), f"expected\n{want}\ngot\n{got}"


def test_stage_two_uses_only_the_first_codebook():
    got = nar_input(TEXT, PROMPT, TARGET, W, 2)
    want = np.array([[0.5, -0.5], [63, 3], [13, 3], [12, 2]], dtype=float)
    assert np.allclose(got, want), f"expected\n{want}\ngot\n{got}"


def test_later_codebooks_do_not_leak():
    a = nar_input(TEXT, PROMPT, TARGET, W, 2)
    changed = TARGET.copy()
    changed[:, 1:] = 0
    b = nar_input(TEXT, PROMPT, changed, W, 2)
    assert np.allclose(a, b), "codebooks j >= i of the target must not affect the stage-i input"


def test_prompt_uses_every_codebook():
    a = nar_input(TEXT, PROMPT, TARGET, W, 2)
    changed = PROMPT.copy()
    changed[:, 2] = 0
    b = nar_input(TEXT, changed, TARGET, W, 2)
    assert not np.allclose(a[1], b[1]), f"the prompt row must change with codebook 3, got {a[1]} and {b[1]}"


def test_order_is_text_prompt_target():
    got = nar_input(TEXT, PROMPT, TARGET, W, 3)
    assert np.allclose(got[0], TEXT[0]), f"row 0 must be the text embedding {TEXT[0]}, got {got[0]}"
    assert np.allclose(got[1], [63, 3]), f"row 1 must be the prompt embedding [63, 3], got {got[1]}"


def test_stage_range():
    for bad in (1, 0, K + 1):
        try:
            nar_input(TEXT, PROMPT, TARGET, W, bad)
        except ValueError:
            continue
        raise AssertionError(f"stage i={bad} must raise ValueError")


def test_ada_layer_norm_identity_scale():
    h = np.array([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]])
    got = ada_layer_norm(h, 1.0, 0.0)
    want = np.array([-1.3416, -0.4472, 0.4472, 1.3416])
    assert np.allclose(got[0], want, atol=1e-3), f"expected {want}, got {got[0]}"
    assert np.allclose(got[1], want, atol=1e-3), f"LayerNorm must ignore input scale; expected {want}, got {got[1]}"


def test_ada_layer_norm_scale_then_shift():
    h = np.array([[1.0, 2.0, 3.0, 4.0]])
    got = ada_layer_norm(h, 2.0, 5.0)
    want = 2.0 * np.array([-1.3416, -0.4472, 0.4472, 1.3416]) + 5.0
    assert np.allclose(got[0], want, atol=1e-3), f"expected a*LN(h)+b = {want}, got {got[0]}"
