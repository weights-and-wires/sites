"""Newton-Schulz on a matrix with known singular values, and on scalars.

Shows that one step maps singular values through phi(x) = a x + b x^3 + c x^5
(blog post, "Proving that NS iteration orthogonalizes the update").
"""
import numpy as np

TUNED = (3.4445, -4.7750, 2.0315)
BASELINE = (2.0, -1.5, 0.5)


def phi(x, coeffs):
    a, b, c = coeffs
    return a * x + b * x**3 + c * x**5


def ns(G, steps, coeffs=TUNED, eps=1e-7):
    a, b, c = coeffs
    X = G / (np.linalg.norm(G) + eps)
    tall = X.shape[0] > X.shape[1]
    if tall:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if tall else X


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    print("phi(1) with tuned coefficients =", round(phi(1.0, TUNED), 4))
    print("phi(1) with baseline coefficients =", round(phi(1.0, BASELINE), 4))
    print("phi'(0) tuned, baseline =", TUNED[0], BASELINE[0])

    print("\nScalar iteration x -> phi(x), 5 steps, tuned coefficients")
    for x0 in [0.001, 0.01, 0.1, 0.5, 1.0]:
        xs = [x0]
        for _ in range(5):
            xs.append(phi(xs[-1], TUNED))
        print(f"x0={x0:<6}", np.array(xs))

    print("\nScalar iteration x -> phi(x), 5 steps, baseline coefficients")
    for x0 in [0.001, 0.01, 0.1, 0.5, 1.0]:
        xs = [x0]
        for _ in range(5):
            xs.append(phi(xs[-1], BASELINE))
        print(f"x0={x0:<6}", np.array(xs))

    # A 4 x 3 matrix with singular values (3, 1, 0.1).
    rng = np.random.default_rng(0)
    U, _ = np.linalg.qr(rng.normal(size=(4, 3)))
    V, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    S = np.array([3.0, 1.0, 0.1])
    G = U @ np.diag(S) @ V.T
    print("\nG has singular values", np.linalg.svd(G, compute_uv=False))
    print("G / ||G||_F has singular values", np.linalg.svd(G / np.linalg.norm(G), compute_uv=False))
    for steps in range(0, 8):
        O = ns(G, steps)
        print(f"after {steps} steps:", np.linalg.svd(O, compute_uv=False))
    O5 = ns(G, 5)
    Uu, Ss, Vt = np.linalg.svd(G, full_matrices=False)
    UVt = Uu @ Vt
    print("\nmax |O5 - U V^T| =", round(float(np.abs(O5 - UVt).max()), 4))
    O_base = ns(G, 25, BASELINE)
    print("baseline coefficients, 25 steps: max |O - U V^T| =", float(f"{np.abs(O_base - UVt).max():.3g}"))
    print("baseline coefficients, 25 steps singular values:", np.linalg.svd(O_base, compute_uv=False))
    print("scale invariance: max |NS(100 G) - NS(G)| =", float(f"{np.abs(ns(100 * G, 5) - O5).max():.3g}"))
