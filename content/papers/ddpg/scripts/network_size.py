"""Numbers for the "Networks" example (Section 7, low-dimensional networks).

Run: python3 scripts/network_size.py
"""


def dense(n_in, n_out):
    return n_in * n_out + n_out


def actor_params(d, n, h1=400, h2=300):
    return dense(d, h1) + dense(h1, h2) + dense(h2, n)


def critic_params(d, n, h1=400, h2=300):
    # The action joins the second hidden layer, so that layer sees h1 + n inputs.
    return dense(d, h1) + dense(h1 + n, h2) + dense(h2, 1)


for d, n in ((3, 1), (17, 6)):
    print(f"state dim {d:>2}, action dim {n}: actor = {actor_params(d, n):,}, critic = {critic_params(d, n):,}")

# The action enters the critic at layer 2: removing the action input changes the count by h2 * n.
print("weights from the action into hidden layer 2 (n = 6):", 300 * 6)
