#!/usr/bin/env python3
"""Compression ratio and the temperature fallback of Section 4.5."""

import numpy as np

from whisper_ref import compression_ratio

THRESH_RATIO, THRESH_LOGPROB = 2.4, -1.0
texts = {
    "normal": "The quick brown fox jumps over the lazy dog while the band plays on.",
    "loop": "Thank you. " * 20,
    "loop short": "Thank you. " * 4,
    "non-Latin": "언덕 위에 올라 내려다보면 너무나 넓고 넓은 들판",
}
for name, t in texts.items():
    b = len(t.encode("utf-8"))
    print(f"{name:10s} chars={len(t):4d} bytes={b:4d} ratio={compression_ratio(t):.3f} fallback={compression_ratio(t) > THRESH_RATIO}")

temps = [round(0.2 * i, 1) for i in range(6)]
print(f"\ntemperature schedule: {temps}")
print(f"np.arange(0, 1.0, 0.2) has {len(np.arange(0, 1.0, 0.2))} values; np.arange(0, 1.0 + 1e-9, 0.2) has {len(np.arange(0, 1.0 + 1e-9, 0.2))}")
print(f"0.2 * 3 = {0.2 * 3!r}; summing 0.2 five times = {sum([0.2] * 5)!r}")

# Table 7 average WER, one row per added heuristic.
table7 = [
    ("Greedy decoding only", 11.0),
    ("+ Beam search", 10.6),
    ("+ Temperature fallback", 10.6),
    ("+ Voice activity detection", 10.2),
    ("+ Previous text conditioning", 10.0),
    ("+ Initial timestamp constraint", 10.0),
]
print("\nTable 7 average WER and relative change from greedy:")
for name, w in table7:
    print(f"  {name:32s} {w:5.1f}  {100 * (w - 11.0) / 11.0:+5.1f}%")
