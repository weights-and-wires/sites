"""Softplus attention gate (Section 2.3, Equation 9) against sigmoid at a few inputs."""
import math


def softplus(x, beta):
    return math.log1p(math.exp(beta * x)) / beta


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


beta = math.log(2)
for x in (-4.0, -1.0, 0.0, 1.0, 4.0):
    print(f"x={x:5.1f}  sigmoid={sigmoid(x):.4f}  softplus(beta=ln2)={softplus(x, beta):.4f}")
# Large positive x: softplus approaches x, so the gate can amplify.
print(f"slope at x=0: {sigmoid(beta * 0):.4f}")
