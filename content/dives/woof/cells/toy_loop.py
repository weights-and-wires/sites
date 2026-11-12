# A TOY autoregressive generator with an end-of-speech stop. It is NOT Woof:
# there are no weights, the "latent" is one number, and the stop signal is
# hand-made. It only mirrors the control flow read from the released runtime.py.
import math
import numpy as np


def logit(p):
    return math.log(p / (1 - p))


def generate(temp=0.5, eos_thresh=0.9, target=18, max_frames=60, min_frames=0, seed=0):
    rng = np.random.default_rng(seed)
    state, frames, eos_log = 0.0, [], []
    eos_t = logit(eos_thresh)
    for f in range(max_frames):
        # toy stop signal: climbs as the frame count passes the "text length"
        e = 0.9 * (f - target) / 2 + 0.3 * state
        eos_log.append(round(e, 3))
        if f > max(3, min_frames) and e > eos_t:  # same test as the runtime
            break
        noise = rng.normal(size=()) * temp
        z = noise + 0.8 * math.sin(0.5 * f) - 0.4 * state  # noise + toy "head"
        frames.append(float(z))
        state = 0.7 * state + 0.5 * z  # the frame is fed back into the state
    return frames, eos_log


for temp in (0.0, 0.5, 1.0):
    frames, _ = generate(temp=temp)
    print(f"temp {temp}: {len(frames)} frames, spread (std) {np.std(frames):.3f}")
for thr in (0.5, 0.9, 0.99):
    frames, _ = generate(eos_thresh=thr)
    print(f"eos threshold {thr}: stops after {len(frames)} frames")
frames, eos_log = generate()
result = {"frames": frames, "eos": eos_log}
