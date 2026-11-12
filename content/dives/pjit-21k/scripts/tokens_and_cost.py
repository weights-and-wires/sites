"""Arithmetic behind the token, bottleneck, EMA, and cost numbers on the page."""
import math

for side, p in ((512, 16), (512, 32), (1024, 32)):
    n = (side // p) ** 2
    print(f"{side}px, {p}px patches: {n} image tokens, {n * n:,} score pairs, {n + 256} tokens with 256 text, {p*p*3} values/patch")
print("pairs ratio 1024 vs 256 tokens:", (1024 / 256) ** 2)
print("with 256 text tokens each: tokens", 1280 / 512, "pairs", (1280 / 512) ** 2)
print("bottleneck ratios for a 32px patch: 3072/1024 =", 3072 / 1024, "; 3072/256 =", 3072 / 256)
print("RoPE interpolation: 1024px patch indices 0..31 times 1/2 ->", 0, "...", 31 / 2)
print("EMA per-step decay for half-life 1954 steps: %.6f" % 0.5 ** (1 / 1954))
print("EMA half-life 1954 steps x 2048 images = %.2fM samples" % (1954 * 2048 / 1e6))
print("v-loss weight: t=0.3 error 0.1 ->", (0.1 / 0.3) ** 2, "; t=0.05 clamped to 0.1 ->", (0.1 / 0.1) ** 2, "; t=0.9 ->", (0.1 / 0.9) ** 2)
print("pyramid: avg pool x=[1,3,5,7,2,2,4,4] by 4 and 2")
x = [1, 3, 5, 7, 2, 2, 4, 4]
print([sum(x[i:i + 4]) / 4 for i in (0, 4)], [sum(x[i:i + 2]) / 2 for i in range(0, 8, 2)])
print("MSE of [3,3] vs [4,3]:", ((3 - 4) ** 2 + 0) / 2)
print("steps:", 610.27e6 / 2048, 49.97e6 / 2048, 80e6 / 2048)
print("costs:", 8139 * 2.5, 276 * 2.5, 8415 * 2.5)
print("params: 2.18 + 0.241 + 0.0028 = %.3fB (this post), 2.18 + 0.145 + 0.00285 = %.3fB (Pyramid-JiT post)" % (2.18 + .241 + .0028, 2.18 + .145 + .00285))
