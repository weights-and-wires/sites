"""What orthogonalizing does to an update, and the Shampoo identity.

Blog post sections: "Why is it good to orthogonalize the update?" and
"Relationship to prior optimizers / Shampoo".
"""
import numpy as np

from ns_example import ns


def ortho(G):
    U, S, Vt = np.linalg.svd(G, full_matrices=False)
    return U @ Vt


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    rng = np.random.default_rng(1)

    # A near-low-rank update: one dominant direction plus small noise.
    n, m = 6, 4
    u, v = rng.normal(size=(n, 1)), rng.normal(size=(m, 1))
    G = 5.0 * u @ v.T + 0.1 * rng.normal(size=(n, m))
    s = np.linalg.svd(G, compute_uv=False)
    print("singular values of G      :", s)
    print("condition number of G     :", round(float(s[0] / s[-1]), 1))
    O = ns(G, 5)
    so = np.linalg.svd(O, compute_uv=False)
    print("singular values of NS5(G) :", so)
    print("condition number of NS5(G):", round(float(so[0] / so[-1]), 2))
    Oex = ortho(G)
    print("exact U V^T singular values:", np.linalg.svd(Oex, compute_uv=False))

    # Step norms: SGD step vs Muon step (Frobenius and spectral).
    print("\nspectral norm of G        :", round(float(s[0]), 3))
    print("Frobenius norm of G       :", round(float(np.linalg.norm(G)), 3))
    print("spectral norm of NS5(G)   :", round(float(so[0]), 3))
    print("Frobenius norm of NS5(G)  :", round(float(np.linalg.norm(O)), 3))
    print("Frobenius norm of U V^T   :", round(float(np.linalg.norm(Oex)), 3), "= sqrt(min(n, m)) =", round(float(np.sqrt(min(n, m))), 3))

    # Share of the squared Frobenius norm carried by the top singular direction.
    print("\nshare of ||G||_F^2 in top direction    :", round(float(s[0] ** 2 / (s**2).sum()), 4))
    print("share of ||UV^T||_F^2 in top direction :", round(1.0 / min(n, m), 4))

    # Shampoo without accumulation equals U V^T (blog post, Shampoo section).
    Gs = rng.normal(size=(5, 3))
    # G G^T is singular when n > m, so use the thin side: (G G^T)^{-1/4} acts on range(U).
    U, S, Vt = np.linalg.svd(Gs, full_matrices=False)
    left = U @ np.diag(S ** -0.5) @ U.T
    right = Vt.T @ np.diag(S ** -0.5) @ Vt
    shampoo = left @ Gs @ right
    print("\nmax |(GG^T)^(-1/4) G (G^TG)^(-1/4) - U V^T| =", float(f"{np.abs(shampoo - ortho(Gs)).max():.3g}"))
