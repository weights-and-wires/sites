"""NAR input embedding sums (Section 4.2.2, Equations 4 and 5), on 3 codebooks."""
import numpy as np

K, V, d = 3, 4, 2
# W^j[v] = [10 * j + v, v], with j starting at 1 as in the paper.
W = np.array([[[10 * (j + 1) + v, v] for v in range(V)] for j in range(K)], dtype=float)

prompt = np.array([[1, 0, 2]])  # one prompt frame, all codebooks
target = np.array([[3, 1, 0], [2, 0, 1]])  # two target frames
i = 3  # predict codebook 3 from codebooks 1 and 2

e_prompt = sum(W[j][prompt[:, j]] for j in range(K))
e_target = sum(W[j][target[:, j]] for j in range(i - 1))
print("prompt embedding (sum over j=1..3):", e_prompt.tolist())
print("target embedding (sum over j<3):", e_target.tolist())
print("stage 2 target embedding:", W[0][target[:, 0]].tolist())
print("input rows:", len(e_prompt) + len(e_target), "(1 prompt + 2 target, text rows omitted)")
