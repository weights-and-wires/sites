"""Chunked prefill: chunk sizes, KV re-reads, and overhead (Section 4.3).

Run from the paper folder: python3 scripts/chunked_prefill.py

Uses the roofline model from roofline_cost.py. The KV size per token assumes
Mistral-7B's shape (32 layers, 8 KV heads, head size 128, fp16), which is the
script's assumption and not a number from the paper.
"""

from roofline_cost import BW, BYTES, PARAMS, PEAK, iteration_time

KV_BYTES_PER_TOKEN = 2 * 8 * 128 * 2 * 32  # K and V, heads, head size, fp16, layers


def chunk_sizes(prompt_len, budget):
    sizes = []
    left = prompt_len
    while left > 0:
        c = min(budget, left)
        sizes.append(c)
        left -= c
    return sizes


def kv_tokens_reread(chunks):
    return sum(sum(chunks[:i]) for i in range(len(chunks)))


if __name__ == "__main__":
    P = 4096
    print(f"prompt of {P} tokens, KV bytes per token = {KV_BYTES_PER_TOKEN}")
    for budget in (512, 1000):
        ch = chunk_sizes(P, budget)
        print(f"budget {budget}: {len(ch)} chunks, last = {ch[-1]}, "
              f"KV tokens re-read = {kv_tokens_reread(ch)}")
    print()
    full, _ = iteration_time(P)
    print(f"unchunked prefill: {full * 1e3:.1f} ms")
    print("chunk  n_chunks  total_ms  weight_overhead  kv_reread_GB  kv_reread_ms")
    for c in (32, 64, 128, 256, 512, 1024, 2048):
        ch = chunk_sizes(P, c)
        total = sum(iteration_time(x)[0] for x in ch)
        kv = kv_tokens_reread(ch) * KV_BYTES_PER_TOKEN
        print(f"{c:5d}  {len(ch):8d}  {total * 1e3:8.1f}  {total / full:15.2f}  "
              f"{kv / 1e9:12.2f}  {kv / BW * 1e3:12.2f}")
