"""Compute the gated activation unit example (Equation 2) and the chain rule (Equation 1).

Run: python3 scripts/gated_example.py
"""

import numpy as np



def causal_conv(x, w, dilation):
    k = len(w)
    y = np.zeros_like(x, dtype=float)
    for j in range(k):
        shift = (k - 1 - j) * dilation
        if shift == 0:
            y += w[j] * x
        else:
            y[shift:] += w[j] * x[:-shift]
    return y


sigmoid = lambda a: 1 / (1 + np.exp(-a))

x = np.array([1.0, 2.0, 3.0, 4.0])
Wf = np.array([1.0, 0.5])
Wg = np.array([0.5, -1.0])
filt = causal_conv(x, Wf, 1)
gate = causal_conv(x, Wg, 1)
r = lambda a: [round(float(v), 4) for v in a]
print("filter conv  =", r(filt))
print("gate conv    =", r(gate))
print("tanh(filter) =", r(np.tanh(filt)))
print("sigmoid(gate)=", r(sigmoid(gate)))
print("z            =", r(np.tanh(filt) * sigmoid(gate)))
print("relu(filter) =", r(np.maximum(filt, 0)))

# Chain rule: a three-sample sequence over three levels.
p1 = np.array([0.5, 0.3, 0.2])            # p(x1)
p2 = {2: np.array([0.1, 0.2, 0.7])}       # p(x2 | x1 = 2)
p3 = {(2, 0): np.array([0.6, 0.3, 0.1])}  # p(x3 | x1 = 2, x2 = 0)
seq = (2, 0, 1)
probs = [p1[seq[0]], p2[2][seq[1]], p3[(2, 0)][seq[2]]]
probs = [float(p) for p in probs]
print("conditionals =", probs)
print("p(x) =", round(float(np.prod(probs)), 6), " log p(x) =", round(float(np.sum(np.log(probs))), 4))
print("sequential steps for 1 s of 16 kHz audio:", 16000)
