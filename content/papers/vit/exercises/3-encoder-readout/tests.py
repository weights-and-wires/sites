import numpy as np
from submission import encoder_block, vit_readout, layernorm, msa, mlp

D, H, F = 8, 2, 16


def make_block(rng, scale=1.0):
    r = lambda *s: rng.normal(size=s) * scale * 0.5
    return dict(h=H, g1=1 + r(D) * 0.1, b1=r(D) * 0.1, g2=1 + r(D) * 0.1, b2=r(D) * 0.1,
                Wq=r(D, D), Wk=r(D, D), Wv=r(D, D), Wo=r(D, D),
                W1=r(D, F), c1=r(F), W2=r(F, D), c2=r(D))


def zero_block():
    z = np.zeros
    return dict(h=H, g1=np.ones(D), b1=z(D), g2=np.ones(D), b2=z(D),
                Wq=z((D, D)), Wk=z((D, D)), Wv=z((D, D)), Wo=z((D, D)),
                W1=z((D, F)), c1=z(F), W2=z((F, D)), c2=z(D))


def test_zero_weights_leave_the_residual_stream_unchanged():
    z = np.random.default_rng(0).normal(size=(5, D)) * 10
    out = encoder_block(z, zero_block())
    assert np.allclose(out, z), (
        "with all sub-block weights 0, each block adds 0 to the residual stream, so the output must equal the input;"
        f" max difference {np.abs(out - z).max():.3g}")


def test_matches_equations_2_and_3():
    rng = np.random.default_rng(1)
    z = rng.normal(size=(5, D)) * 3
    p = make_block(rng)
    zp = z + msa(layernorm(z, p["g1"], p["b1"]), p)
    want = zp + mlp(layernorm(zp, p["g2"], p["b2"]), p)
    got = encoder_block(z, p)
    assert np.allclose(got, want), f"expected\n{want}\ngot\n{got}"


def test_residual_adds_to_the_unnormalized_input():
    rng = np.random.default_rng(2)
    p = make_block(rng)
    z = np.vstack([np.full(D, 100.0) + rng.normal(size=D), rng.normal(size=(3, D))])
    out = encoder_block(z, p)
    assert np.abs(out[0]).max() > 50, f"the large input must survive on the identity path, got row 0 = {out[0]}"


def test_readout_is_layernorm_of_class_row_only():
    rng = np.random.default_rng(3)
    blocks = [make_block(rng), make_block(rng)]
    z0 = rng.normal(size=(6, D))
    g, b = 1 + rng.normal(size=D) * 0.1, rng.normal(size=D) * 0.1
    y = vit_readout(z0, blocks, g, b)
    z = z0
    for p in blocks:
        z = encoder_block(z, p)
    want = layernorm(z[0], g, b)
    assert y.shape == (D,), f"expected shape ({D},), got {y.shape}"
    assert np.allclose(y, want), f"expected {want}, got {y}"


def test_final_layernorm_is_applied():
    rng = np.random.default_rng(4)
    z0 = rng.normal(size=(4, D)) * 50
    y = vit_readout(z0, [zero_block()], np.ones(D), np.zeros(D))
    assert abs(y.mean()) < 1e-6 and abs(y.std() - 1) < 1e-3, f"y must have mean 0 and std 1 with g=1, b=0; got mean {y.mean():.3g}, std {y.std():.3g}"


def test_patch_order_does_not_matter_without_position_embeddings():
    rng = np.random.default_rng(5)
    blocks = [make_block(rng), make_block(rng)]
    cls, patches = rng.normal(size=(1, D)), rng.normal(size=(6, D))
    perm = rng.permutation(6)
    g, b = np.ones(D), np.zeros(D)
    a = vit_readout(np.vstack([cls, patches]), blocks, g, b)
    c = vit_readout(np.vstack([cls, patches[perm]]), blocks, g, b)
    assert np.allclose(a, c, atol=1e-8), f"shuffling patches must not change the class readout; max diff {np.abs(a - c).max():.3g}"
    pos = rng.normal(size=(7, D))
    a = vit_readout(np.vstack([cls, patches]) + pos, blocks, g, b)
    c = vit_readout(np.vstack([cls, patches[perm]]) + pos, blocks, g, b)
    assert not np.allclose(a, c, atol=1e-3), "with position embeddings added, shuffling the patches must change the readout"
