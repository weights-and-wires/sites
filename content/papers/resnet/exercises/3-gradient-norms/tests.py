import numpy as np
from submission import input_grad_norms, layer_forward, make_net


def loss_from(net, h, residual, start=0):
    for layer in net[start:]:
        h, _ = layer_forward(layer, h, residual)
    return 0.5 * float(h @ h)


def numeric_grad_norm(net, h, residual, start):
    eps = 1e-6
    g = np.zeros_like(h)
    for i in range(len(h)):
        e = np.zeros_like(h)
        e[i] = eps
        g[i] = (loss_from(net, h + e, residual, start) - loss_from(net, h - e, residual, start)) / (2 * eps)
    return np.linalg.norm(g)


def test_length_and_last_entry():
    for residual in (False, True):
        net = make_net(6, 8, 0, residual)
        x = np.random.default_rng(1).normal(size=8)
        norms = np.asarray(input_grad_norms(net, x, residual))
        assert norms.shape == (len(net) + 1,), f"residual={residual}: expected shape {(len(net) + 1,)}, got {norms.shape}"
        h = x
        for layer in net:
            h, _ = layer_forward(layer, h, residual)
        assert np.isclose(norms[-1], np.linalg.norm(h)), f"residual={residual}: the last entry is ||dL/dh_n|| = ||h_n|| = {np.linalg.norm(h)}, got {norms[-1]}"


def test_input_gradient_matches_finite_differences():
    for residual in (False, True):
        net = make_net(6, 8, 3, residual)
        x = np.random.default_rng(2).normal(size=8)
        got = input_grad_norms(net, x, residual)[0]
        want = numeric_grad_norm(net, x, residual, 0)
        assert np.isclose(got, want, rtol=1e-4), f"residual={residual}: expected ||dL/dx|| = {want}, got {got}"


def test_middle_entry_matches_finite_differences():
    for residual in (False, True):
        net = make_net(8, 8, 4, residual)
        x = np.random.default_rng(5).normal(size=8)
        l = 2
        h = x
        for layer in net[:l]:
            h, _ = layer_forward(layer, h, residual)
        got = input_grad_norms(net, x, residual)[l]
        want = numeric_grad_norm(net, h, residual, l)
        assert np.isclose(got, want, rtol=1e-4), f"residual={residual}: expected ||dL/dh_{l}|| = {want}, got {got} (check the order of the entries)"


def test_plain_net_gradient_vanishes():
    for seed in range(3):
        x = np.random.default_rng(100 + seed).normal(size=32)
        n = input_grad_norms(make_net(20, 32, seed, False), x, False)
        ratio = n[0] / n[-1]
        assert ratio < 1e-3, f"seed {seed}: the plain 20-layer net should have ||g_0|| / ||g_20|| < 1e-3, got {ratio:.3g}"


def test_residual_net_gradient_survives():
    for seed in range(3):
        x = np.random.default_rng(100 + seed).normal(size=32)
        n = input_grad_norms(make_net(20, 32, seed, True), x, True)
        ratio = n[0] / n[-1]
        assert ratio > 0.5, f"seed {seed}: the residual net should keep ||g_0|| / ||g_10|| above 0.5, got {ratio:.3g}"
