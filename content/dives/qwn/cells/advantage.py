# Group-relative advantages for one prompt's group of answers. Edit the
# (coherence, responds) pairs and run again. These pairs are our example.
import numpy as np

scores = np.array([(5, 5), (4, 4), (5, 3), (2, 2)])


def advantages(r):
    return (r - r.mean()) / (r.std() + 1e-6)   # population std, as in the page


summed = (scores[:, 0] - 1 + scores[:, 1] - 1) / 8
lowest = (scores.min(axis=1) - 1) / 4
usable = (scores >= 4).all(axis=1)

for name, r in [("sum", summed), ("min", lowest)]:
    print(name, "rewards   ", np.round(r, 2).tolist())
    print(name, "advantages", np.round(advantages(r), 2).tolist())
print("usable        ", usable.tolist())
