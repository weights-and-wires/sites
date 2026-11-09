import math


def forward_grid(batch, heads, N, Br, parallel_seq):
    """Thread blocks launched by the forward pass.

    FlashAttention uses one block per (batch, head). FlashAttention-2 adds one
    block per row block of Q (Section 3.2).
    """
    blocks = batch * heads
    if parallel_seq:
        blocks *= math.ceil(N / Br)
    return blocks


def backward_grid(batch, heads, N, Bc):
    """FlashAttention-2 backward pass: one block per column block of K and V."""
    return batch * heads * math.ceil(N / Bc)


def waves(blocks, sms=108):
    """Rounds needed if each SM runs one block at a time."""
    return math.ceil(blocks / sms)


def sm_utilization(blocks, sms=108):
    """Fraction of SM-time that holds a block, over all waves."""
    return blocks / (waves(blocks, sms) * sms)
