def pretrain_k_hours(k_per_trillion, tokens_trillion):
    """Pre-training GPU hours, in thousands."""
    return k_per_trillion * tokens_trillion


def days_per_trillion(k_per_trillion, gpus):
    """Wall-clock days for one trillion tokens on `gpus` GPUs."""
    return k_per_trillion * 1000 / gpus / 24


def total_k_hours(stages):
    """Sum of the stage hours (thousands)."""
    return sum(stages)


def cost_million_usd(k_hours, price):
    """Cost in millions of dollars for `k_hours` thousand GPU hours at `price` per hour."""
    return k_hours * 1000 * price / 1e6


def stage_shares(stages):
    """Fraction of the total hours spent in each stage."""
    total = sum(stages)
    return [s / total for s in stages]
