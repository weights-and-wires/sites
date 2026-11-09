"""Loss of the toy network from exercise 4 under Muon and AdamW on the hidden matrix.

Runs the exercise solution, so it needs no other code. Learning rates are the
exercise defaults (lr_muon = 0.02, lr_adam = 0.01, mu = 0.95), not tuned.
"""
import importlib.util
import pathlib

import numpy as np

path = pathlib.Path(__file__).resolve().parents[1] / "exercises/4-hybrid-training/solution.py"
spec = importlib.util.spec_from_file_location("solution", path)
sol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sol)

if __name__ == "__main__":
    for steps in (60, 200):
        print(f"steps = {steps}")
        for use_muon in (True, False):
            out = sol.train(use_muon, steps=steps)
            l = out["losses"]
            name = "Muon (W_h) + AdamW" if use_muon else "AdamW everywhere"
            print(f"  {name:<20} loss at step 1, 20, {steps}: {l[0]:.4f}, {l[19]:.4f}, {l[-1]:.2e}")
    d = sol.train(True, steps=1)["deltas"]["W_h"] / 0.02
    s = np.linalg.svd(d, compute_uv=False)
    print("first Muon step on W_h / lr: top, bottom singular value =", f"{s[0]:.3f}, {s[-1]:.3f}")
    d = sol.train(False, steps=1)["deltas"]["W_h"] / 0.01
    s = np.linalg.svd(d, compute_uv=False)
    print("first AdamW step on W_h / lr: top singular value =", f"{s[0]:.2f}", "; largest entry =", f"{np.abs(d).max():.2f}")
