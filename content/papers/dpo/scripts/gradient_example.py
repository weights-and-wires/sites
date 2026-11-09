"""Compute the numbers for the DPO gradient example (Section 4, Appendix A.4).

The policy is a softmax over three responses to one prompt, with logits theta.

Run: python3 content/papers/dpo/scripts/gradient_example.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def loss(theta, ref_logits, w, l, beta):
    lp, lr = np.log(softmax(theta)), np.log(softmax(ref_logits))
    margin = beta * ((lp[w] - lr[w]) - (lp[l] - lr[l]))
    return float(np.logaddexp(0.0, -margin))


def grad(theta, ref_logits, w, l, beta):
    lp, lr = np.log(softmax(theta)), np.log(softmax(ref_logits))
    rhat = beta * (lp - lr)
    weight = sigmoid(rhat[l] - rhat[w])
    e_w, e_l = np.eye(3)[w], np.eye(3)[l]
    return -beta * weight * (e_w - e_l), weight


ref_logits = np.log(np.array([0.70, 0.25, 0.05]))
w, l, beta = 2, 0, 0.5  # response 2 preferred over response 0

g, weight = grad(ref_logits, ref_logits, w, l, beta)
print("at pi = pi_ref: weight sigma(rhat_l - rhat_w) =", round(float(weight), 4))
print("at pi = pi_ref: gradient wrt theta           =", g.tolist())

eps = 1e-6
fd = np.zeros(3)
for k in range(3):
    t1, t2 = ref_logits.copy(), ref_logits.copy()
    t1[k] += eps
    t2[k] -= eps
    fd[k] = (loss(t1, ref_logits, w, l, beta) - loss(t2, ref_logits, w, l, beta)) / (2 * eps)
print("finite-difference gradient                   =", fd.tolist())

print()
print("Weight when the implicit reward orders the pair wrongly or rightly")
for name, theta in (
    ("wrong (policy moved toward the loser)", ref_logits + np.array([2.0, 0.0, -2.0])),
    ("right (policy moved toward the winner)", ref_logits + np.array([-2.0, 0.0, 2.0])),
):
    gg, ww = grad(theta, ref_logits, w, l, beta)
    p = softmax(theta)
    rhat = beta * (np.log(p) - np.log(softmax(ref_logits)))
    print(f"  {name}")
    print("    rhat_w - rhat_l =", round(float(rhat[w] - rhat[l]), 4), " weight =", round(float(ww), 4), " loss =", round(loss(theta, ref_logits, w, l, beta), 4))

print()
print("Ten gradient-descent steps with learning rate 1.0")
theta = ref_logits.copy()
for step in range(1, 11):
    g, weight = grad(theta, ref_logits, w, l, beta)
    theta = theta - 1.0 * g
    if step in (1, 5, 10):
        p = softmax(theta)
        print(f"  step {step:2d}: weight before step = {weight:.4f}, pi = {p.tolist()}, loss = {loss(theta, ref_logits, w, l, beta):.4f}")
