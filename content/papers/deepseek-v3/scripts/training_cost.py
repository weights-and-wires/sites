"""Training cost arithmetic from Table 1 and Section 1 of the DeepSeek-V3 report."""

PRICE = 2.0  # USD per H800 GPU hour (Table 1 assumption)
GPUS = 2048
PER_TRILLION_K = 180  # thousand GPU hours per trillion tokens (Section 1)
TOKENS_T = 14.8
STAGES = {"pre-training": 2664, "context extension": 119, "post-training": 5}  # thousand GPU hours


def main():
    pre = PER_TRILLION_K * TOKENS_T
    print(f"pre-training: {PER_TRILLION_K}K x {TOKENS_T} = {pre:.0f}K GPU hours (Table 1 lists 2664K)")
    days_per_t = PER_TRILLION_K * 1000 / GPUS / 24
    print(f"days per trillion tokens on {GPUS} GPUs: {days_per_t:.2f} (Section 1 says 3.7)")
    print(f"pre-training days: {2664 * 1000 / GPUS / 24:.1f} (about {2664 * 1000 / GPUS / 24 / 30:.1f} months)")
    total = sum(STAGES.values())
    print(f"total = {' + '.join(str(v) for v in STAGES.values())} = {total}K GPU hours (Table 1 lists 2788K)")
    for name, k in STAGES.items():
        print(f"  {name:18s} {k:5d}K hours -> ${k * 1000 * PRICE / 1e6:.3f}M, {k / total * 100:.1f}% of total")
    print(f"total cost at ${PRICE:.0f} per GPU hour: ${total * 1000 * PRICE / 1e6:.3f}M (Table 1 lists $5.576M)")
    print(f"pre-training GPU-milliseconds per token: {2664 * 1000 * 3600 * 1000 / (TOKENS_T * 1e12):.3f}")
    print(f"cost per trillion tokens at ${PRICE:.0f}/hour: ${PER_TRILLION_K * 1000 * PRICE / 1e6:.2f}M")
    # a change in price
    for price in (1.0, 2.0, 3.0):
        print(f"  price ${price:.0f}/hour -> total ${total * 1000 * price / 1e6:.3f}M")


if __name__ == "__main__":
    main()
