def pretrain_k_hours(k_per_trillion, tokens_trillion):
    """Pre-training GPU hours, in thousands."""
    raise NotImplementedError


def days_per_trillion(k_per_trillion, gpus):
    """Wall-clock days for one trillion tokens on `gpus` GPUs."""
    raise NotImplementedError


def total_k_hours(stages):
    """Sum of the stage hours (thousands)."""
    raise NotImplementedError


def cost_million_usd(k_hours, price):
    """Cost in millions of dollars for `k_hours` thousand GPU hours at `price` per hour."""
    raise NotImplementedError


def stage_shares(stages):
    """Fraction of the total hours spent in each stage."""
    raise NotImplementedError
