"""Build the stage 2 sequence of Section III-C on a tiny example.

Run: python3 scripts/stage_sequences.py
"""

import numpy as np

K, N, Q, QP = 8, 16, 3, 2  # tiny sizes, so the ids are readable
z = np.array([3, 3, 5])    # T_S = 3 semantic tokens
Y = np.array([[1, 2, 9], [4, 5, 9], [6, 7, 9], [8, 9, 9], [10, 11, 9], [12, 13, 9]])  # T_A = 6
print("T_S =", len(z), " T_A =", len(Y), " (T_A = 2 T_S)")

coarse = Y[:, :QP]
acoustic = (coarse + np.arange(QP) * N + K).reshape(-1)
seq = np.concatenate([z, acoustic])
mask = np.concatenate([np.zeros(len(z), bool), np.ones(len(acoustic), bool)])
print("coarse acoustic ids =", acoustic.tolist())
print("stage 2 sequence    =", seq.tolist())
print("loss mask           =", mask.astype(int).tolist())
print("sequence length     =", len(seq), "= T_S + T_A * Q' =", len(z) + len(Y) * QP)
print("predicted tokens    =", int(mask.sum()))

print("\nRemoving consecutive repeats from semantic tokens (Section IV-B)")
runs = np.array([4, 4, 4, 7, 7, 4, 9])
keep = np.concatenate([[True], runs[1:] != runs[:-1]])
print("tokens      =", runs.tolist())
print("collapsed   =", runs[keep].tolist())
idx = np.flatnonzero(keep)
lengths = np.diff(np.append(idx, len(runs)))
print("run lengths =", lengths.tolist())
