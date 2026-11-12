"""Evaluate the soft overlong punishment (Section 3.4, Equation 13) with the
constants from Section 4.1: L_max - L_cache = 16384 and L_cache = 4096.

Run: python3 scripts/overlong_example.py
"""

L_CACHE = 4096
L_EXPECTED = 16384          # the paper's "expected maximum length"
L_MAX = L_EXPECTED + L_CACHE  # hard generation limit, 20480


def r_length(n, l_max=L_MAX, l_cache=L_CACHE):
    if n <= l_max - l_cache:
        return 0.0
    if n <= l_max:
        return ((l_max - l_cache) - n) / l_cache
    return -1.0


print(f"hard generation limit = {L_MAX}, punishment starts at {L_MAX - L_CACHE}")
print("length   R_length")
for n in (8000, 16384, 18432, 20000, 20480, 25000):
    print(f"{n:<7}  {r_length(n):+.4f}")

print()
print("Total reward = R(y_hat, y) + R_length(y) with R in {+1, -1} (Equation 7)")
for correct, n in ((True, 16000), (True, 18432), (True, 20480), (False, 18432)):
    base = 1.0 if correct else -1.0
    print(f"correct={correct!s:<5} length={n:<6} total = {base + r_length(n):+.4f}")
