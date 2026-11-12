"""KV cache transfer between a prefill and a decoding instance (Section 3.3, "Communication overhead").

Run from the paper folder: python3 scripts/kv_transfer.py

A token's KV cache holds one key vector and one value vector of the hidden size
per layer. OPT-66B has 64 layers and hidden size 9216 (the OPT paper's
configuration, not a number from the DistServe paper). Weights and the cache are
fp16, as in the paper's experiments (Section 6.1).
"""

GIB = 2**30


def kv_bytes_per_token(layers, hidden, bytes_per_value=2):
    """2 (key and value) x layers x hidden x bytes."""
    return 2 * layers * hidden * bytes_per_value


def transfer_seconds(n_bytes, bandwidth_bytes_per_s):
    return n_bytes / bandwidth_bytes_per_s


def needed_gbit_per_s(n_bytes_per_request, rate):
    """Bandwidth in Gbit/s (10^9 bits) that moves `rate` requests per second."""
    return n_bytes_per_request * rate * 8 / 1e9


if __name__ == "__main__":
    per_token = kv_bytes_per_token(64, 9216)
    request = 512 * per_token
    print(f"bytes per token: {per_token}")
    print(f"bytes per 512-token request: {request} = {request / GIB:.3f} GiB = {request / 1e9:.3f} GB")
    print(f"at 10 requests/s: {10 * request / GIB:.2f} GiB/s = {needed_gbit_per_s(request, 10):.1f} Gbit/s"
          f" = {10 * request * 8 / GIB:.1f} Gibit/s")
    links = (
        ("NVLink 600 GB/s (the paper's A100 peak)", 600e9),
        ("NVLink 300 GB/s (one direction)", 300e9),
        ("InfiniBand 400 Gbit/s", 400e9 / 8),
        ("Ethernet 25 Gbit/s (the paper's testbed)", 25e9 / 8),
    )
    print("transfer time of one 512-token OPT-66B request")
    for name, bw in links:
        print(f"  {name}: {transfer_seconds(request, bw) * 1e3:.2f} ms")
    mha = kv_bytes_per_token(64, 9216)
    gqa = kv_bytes_per_token(64, 9216 // 8)
    print(f"grouped-query attention with 8 groups cuts the cache {mha // gqa}x: {mha} -> {gqa} bytes per token")
