import torch
from submission import Block, train

torch.set_num_threads(1)


def test_block_keeps_shape():
    torch.manual_seed(0)
    block = Block(64, 4)
    x = torch.randn(5, 3, 64)
    y = block(x)
    assert y.shape == x.shape, f"expected shape {tuple(x.shape)}, got {tuple(y.shape)}"


def test_block_is_pre_ln_residual():
    # With every parameter set to zero, both sub-layers output zero, so a pre-LN block
    # returns x. A post-LN block returns LayerNorm(x), and a block without residuals returns 0.
    torch.manual_seed(0)
    block = Block(64, 4)
    with torch.no_grad():
        for p in block.parameters():
            p.zero_()
    x = torch.randn(2, 3, 64) * 3 + 1
    y = block(x)
    assert torch.allclose(y, x, atol=1e-5), f"with zeroed weights the block must return x; max difference {(y - x).abs().max().item():.3g}"


def test_positions_mix():
    # Attention lets the last token read the first token.
    torch.manual_seed(0)
    block = Block(64, 4).eval()
    x = torch.randn(1, 3, 64)
    x2 = x.clone()
    x2[0, 0] += torch.randn(64)
    with torch.no_grad():
        d = (block(x)[0, 2] - block(x2)[0, 2]).abs().max().item()
    assert d > 1e-6, "changing token 0 did not change the output at position 2; attention is not mixing positions"


def test_has_four_head_attention_parameters():
    block = Block(64, 4)
    n = sum(p.numel() for p in block.parameters())
    # Q, K, V, and W^O: 4 * (64 * 64 + 64); FFN: 64 * 256 + 256 + 256 * 64 + 64; two LayerNorms: 4 * 64.
    want = 4 * (64 * 64 + 64) + (64 * 256 + 256 + 256 * 64 + 64) + 4 * 64
    assert n == want, f"expected {want} parameters (attention, FFN with 4 * d_model inner width, two LayerNorms), got {n}"


def test_learns_modular_addition():
    acc = train(seed=0, steps=5000)
    assert acc >= 0.9, f"validation accuracy {acc:.3f} is below 0.9 within 5,000 steps"
