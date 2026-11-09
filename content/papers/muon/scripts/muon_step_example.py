"""Two Muon steps on a 2 x 2 matrix, and momentum before vs after orthogonalizing.

Blog post, "Definition" (Algorithm 2) and "Orthogonal-SGDM".
"""
import numpy as np

from update_geometry import ortho

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    lr, mu = 0.1, 0.9
    W = np.zeros((2, 2))
    G1 = np.array([[3.0, 0.0], [0.0, 1.0]])
    G2 = np.array([[0.0, 1.0], [-1.0, 0.0]])

    B = np.zeros((2, 2))
    print("Muon with exact orthogonalization (Algorithm 2)")
    for t, G in enumerate([G1, G2], start=1):
        B = mu * B + G
        O = ortho(B)
        W = W - lr * O
        print(f"t={t}: B =\n{B}\n      O = U V^T =\n{O}\n      W =\n{W}")
    muon_W = W

    # Orthogonal-SGDM: orthogonalize the gradient, then apply momentum.
    W = np.zeros((2, 2))
    M = np.zeros((2, 2))
    for G in [G1, G2]:
        M = mu * M + ortho(G)
        W = W - lr * M
    print("\nOrthogonal-SGDM (orthogonalize first, then momentum): W =\n", W)
    print("difference from Muon, max abs:", round(float(np.abs(W - muon_W).max()), 4))

    # Singular values of each variant's second step.
    B2 = mu * G1 + G2
    print("\nsingular values of B_2 =", np.linalg.svd(B2, compute_uv=False))
    print("singular values of the Muon step direction:", np.linalg.svd(ortho(B2), compute_uv=False))
    print("singular values of the Orthogonal-SGDM step direction:", np.linalg.svd(mu * ortho(G1) + ortho(G2), compute_uv=False))
