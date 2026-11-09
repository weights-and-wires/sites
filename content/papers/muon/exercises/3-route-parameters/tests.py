import numpy as np
from submission import route, flatten_for_muon


def test_hidden_matrices_go_to_muon():
    for name, shape in [("blocks.0.attn.wq", (64, 64)), ("blocks.0.attn.wk", (64, 64)), ("blocks.3.mlp.fc1.weight", (256, 64)), ("blocks.3.mlp.fc2.weight", (64, 256))]:
        got = route(name, shape)
        assert got == "muon", f"{name} {shape}: expected 'muon', got {got!r}"


def test_embeddings_and_head_go_to_adamw_even_though_they_are_2d():
    for name, shape in [("tok_emb.weight", (1000, 64)), ("pos_emb.weight", (128, 64)), ("lm_head.weight", (1000, 64)), ("wte.weight", (1000, 64))]:
        got = route(name, shape)
        assert got == "adamw", f"{name} {shape}: expected 'adamw', got {got!r}"


def test_vectors_and_scalars_go_to_adamw():
    for name, shape in [("blocks.0.ln1.weight", (64,)), ("blocks.0.ln1.bias", (64,)), ("blocks.0.mlp.fc1.bias", (256,)), ("logit_scale", ())]:
        got = route(name, shape)
        assert got == "adamw", f"{name} {shape}: expected 'adamw', got {got!r}"


def test_prefix_not_substring():
    got = route("blocks.2.attn.head_mix.weight", (8, 8))
    assert got == "muon", f"a hidden matrix whose name contains 'head' after a dot must go to 'muon', got {got!r}"


def test_conv_filters_go_to_muon_and_other_ranks_do_not():
    assert route("stem.conv.weight", (16, 3, 3, 3)) == "muon", f"4D filter: expected 'muon', got {route('stem.conv.weight', (16, 3, 3, 3))!r}"
    got = route("odd.weight", (4, 4, 4))
    assert got == "adamw", f"a 3D parameter: expected 'adamw', got {got!r}"


def test_flatten_conv_shape_and_order():
    G = np.arange(2 * 3 * 2 * 2, dtype=float).reshape(2, 3, 2, 2)
    out = flatten_for_muon(G)
    assert out.shape == (2, 12), f"expected shape (2, 12), got {out.shape}"
    assert np.array_equal(out[1], G[1].ravel()), f"row 1 must be the whole second filter {G[1].ravel()}, got {out[1]}"


def test_flatten_leaves_matrices_alone():
    G = np.arange(6, dtype=float).reshape(2, 3)
    out = flatten_for_muon(G)
    assert out.shape == (2, 3) and np.array_equal(out, G), f"a 2D array must come back unchanged, got\n{out}"
