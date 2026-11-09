"""A 5x5 gridworld and a tiny numpy DQN, shared by the page's scripts.

The agent starts in the top-left corner and gets reward +1 on reaching the
bottom-right corner, which ends the episode. Every other step gives 0.
Run: python3 scripts/gridworld.py   (self-check of the environment)
"""

import numpy as np

SIZE = 5
N_ACTIONS = 4  # up, down, left, right
MOVES = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
GAMMA = 0.9
MAX_STEPS = 50


def reset():
    return (0, 0)


def step(pos, a):
    r, c = np.clip(np.array(pos) + MOVES[a], 0, SIZE - 1)
    pos = (int(r), int(c))
    done = pos == (SIZE - 1, SIZE - 1)
    return pos, (1.0 if done else 0.0), done


def features(pos):
    """Two coordinates scaled to [0, 1]: the network must generalize across cells."""
    return np.array(pos, dtype=float) / (SIZE - 1)


def init_params(rng, hidden=32):
    return {
        "W1": rng.normal(0, 1, (2, hidden)) * np.sqrt(2 / 2),
        "b1": np.zeros(hidden),
        "W2": rng.normal(0, 1, (hidden, N_ACTIONS)) * np.sqrt(2 / hidden),
        "b2": np.zeros(N_ACTIONS),
    }


def q_values(p, s):
    h = np.maximum(0, s @ p["W1"] + p["b1"])
    return h @ p["W2"] + p["b2"]


def sgd_step(p, s, a, r, s2, done, gamma, lr):
    """One minibatch step of Equation 3. Returns the mean squared error before the step."""
    y = r + gamma * (1 - done) * q_values(p, s2).max(axis=1)
    z1 = s @ p["W1"] + p["b1"]
    h = np.maximum(0, z1)
    q = h @ p["W2"] + p["b2"]
    B = len(a)
    idx = np.arange(B)
    err = q[idx, a] - y
    dq = np.zeros_like(q)
    dq[idx, a] = 2 * err / B
    dh = dq @ p["W2"].T * (z1 > 0)
    p["W2"] -= lr * h.T @ dq
    p["b2"] -= lr * dq.sum(0)
    p["W1"] -= lr * s.T @ dh
    p["b1"] -= lr * dh.sum(0)
    return float((err**2).mean())


def greedy_success(p):
    pos = reset()
    for _ in range(MAX_STEPS):
        a = int(np.argmax(q_values(p, features(pos))))
        pos, _, done = step(pos, a)
        if done:
            return True
    return False


def train(seed, steps=4000, replay=True, batch=32, lr=0.05, eps_steps=1500):
    rng = np.random.default_rng(seed)
    p = init_params(rng)
    mem, cap = [], 2000
    pos = reset()
    t_ep = 0
    returns, losses = [], []
    for t in range(steps):
        eps = max(0.1, 1 - 0.9 * t / eps_steps)
        s = features(pos)
        a = int(rng.integers(N_ACTIONS)) if rng.random() < eps else int(np.argmax(q_values(p, s)))
        pos2, r, done = step(pos, a)
        t_ep += 1
        tr = (s, a, r, features(pos2), float(done))
        if replay:
            if len(mem) < cap:
                mem.append(tr)
            else:
                mem[t % cap] = tr
            if len(mem) >= batch:
                idx = rng.integers(len(mem), size=batch)
                b = [mem[i] for i in idx]
            else:
                b = None
        else:
            b = [tr]
        if b is not None:
            S, A, R, S2, D = (np.array(x) for x in zip(*b))
            losses.append(sgd_step(p, S, A.astype(int), R, S2, D, GAMMA, lr))
        pos = pos2
        if done or t_ep >= MAX_STEPS:
            returns.append(float(done))
            pos, t_ep = reset(), 0
    return p, returns, losses


if __name__ == "__main__":
    pos = reset()
    for a in [1, 1, 1, 1, 3, 3, 3, 3]:
        pos, r, done = step(pos, a)
    print("after 8 moves:", pos, r, done)
