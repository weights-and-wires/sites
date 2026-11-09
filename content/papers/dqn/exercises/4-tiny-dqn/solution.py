import numpy as np

SIZE = 5
N_ACTIONS = 4  # up, down, left, right
MOVES = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
GAMMA = 0.9
MAX_STEPS = 50


def reset():
    return (0, 0)


def step(pos, a):
    """Move, staying inside the grid. Reward 1 and episode end at the bottom-right cell."""
    r, c = np.clip(np.array(pos) + MOVES[a], 0, SIZE - 1)
    pos = (int(r), int(c))
    done = pos == (SIZE - 1, SIZE - 1)
    return pos, (1.0 if done else 0.0), done


def features(pos):
    return np.array(pos, dtype=float) / (SIZE - 1)


def init_params(rng, hidden=32):
    return {
        "W1": rng.normal(0, 1, (2, hidden)),
        "b1": np.zeros(hidden),
        "W2": rng.normal(0, 1, (hidden, N_ACTIONS)) * np.sqrt(2 / hidden),
        "b2": np.zeros(N_ACTIONS),
    }


def q_values(p, S):
    """Q-values of shape (B, N_ACTIONS) for states S of shape (B, 2)."""
    return np.maximum(0, S @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"]


def dqn_step(p, S, A, R, S2, D, gamma, lr):
    """One minibatch step of deep Q-learning. Updates p in place; returns the loss before the step."""
    B = len(A)
    idx = np.arange(B)
    y = R + gamma * (1.0 - D) * q_values(p, S2).max(axis=1)
    z1 = S @ p["W1"] + p["b1"]
    h = np.maximum(0, z1)
    q = h @ p["W2"] + p["b2"]
    err = q[idx, A] - y
    dq = np.zeros_like(q)
    dq[idx, A] = 2 * err / B
    dh = (dq @ p["W2"].T) * (z1 > 0)
    grads = {"W2": h.T @ dq, "b2": dq.sum(0), "W1": S.T @ dh, "b1": dh.sum(0)}
    for k, g in grads.items():
        p[k] -= lr * g
    return float(np.mean(err**2))


def greedy_reaches_goal(p):
    pos = reset()
    for _ in range(MAX_STEPS):
        a = int(np.argmax(q_values(p, features(pos)[None])[0]))
        pos, _, done = step(pos, a)
        if done:
            return True
    return False


def train_dqn(seed, steps=4000, lr=0.05, batch=32, capacity=2000, eps_steps=1500, use_replay=True):
    """Finished for you: epsilon-greedy play, a replay memory, and one dqn_step per environment step."""
    rng = np.random.default_rng(seed)
    p = init_params(rng)
    memory = []
    pos, t_ep = reset(), 0
    for t in range(steps):
        eps = max(0.1, 1 - 0.9 * t / eps_steps)
        s = features(pos)
        if rng.random() < eps:
            a = int(rng.integers(N_ACTIONS))
        else:
            a = int(np.argmax(q_values(p, s[None])[0]))
        pos2, r, done = step(pos, a)
        tr = (s, a, r, features(pos2), float(done))
        if use_replay:
            if len(memory) < capacity:
                memory.append(tr)
            else:
                memory[t % capacity] = tr
            if len(memory) >= batch:
                batch_tr = [memory[i] for i in rng.integers(len(memory), size=batch)]
            else:
                batch_tr = None
        else:
            batch_tr = [tr]
        if batch_tr is not None:
            S, A, R, S2, D = (np.array(x) for x in zip(*batch_tr))
            dqn_step(p, S, A.astype(int), R, S2, D, GAMMA, lr)
        t_ep += 1
        pos = pos2
        if done or t_ep >= MAX_STEPS:
            pos, t_ep = reset(), 0
    return p
