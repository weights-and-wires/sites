#!/usr/bin/env python3
"""Check the arithmetic in Table 2: relative error reduction (RER) of Whisper Large V2
against wav2vec 2.0 Large (no LM), which both score 2.7 on LibriSpeech Clean."""

import numpy as np

rows = {  # dataset: (wav2vec 2.0 Large, Whisper Large V2, RER % as printed)
    "LibriSpeech Clean": (2.7, 2.7, 0.0),
    "Artie": (24.5, 6.2, 74.7),
    "Common Voice": (29.9, 9.0, 69.9),
    "Fleurs En": (14.6, 4.4, 69.9),
    "Tedlium": (10.5, 4.0, 61.9),
    "CHiME6": (65.8, 25.5, 61.2),
    "VoxPopuli En": (17.9, 7.3, 59.2),
    "CORAAL": (35.6, 16.2, 54.5),
    "AMI IHM": (37.0, 16.9, 54.3),
    "Switchboard": (28.3, 13.8, 51.2),
    "CallHome": (34.8, 17.6, 49.4),
    "WSJ": (7.7, 3.9, 49.4),
    "AMI SDM1": (67.6, 36.4, 46.2),
    "LibriSpeech Other": (6.2, 5.2, 16.1),
}
rer = []
for name, (w2v, wh, printed) in rows.items():
    r = 100 * (w2v - wh) / w2v
    rer.append(r)
    flag = "" if abs(r - printed) < 0.15 else "  <- differs from the printed value"
    print(f"{name:18s} {w2v:5.1f} {wh:5.1f} RER {r:5.1f} (printed {printed:4.1f}){flag}")
a = np.array([v[0] for v in rows.values()])[1:]  # the 13 datasets other than LibriSpeech Clean
b = np.array([v[1] for v in rows.values()])[1:]
print(f"\n13 out-of-distribution datasets: average WER wav2vec 2.0 {a.mean():.1f}, Whisper {b.mean():.1f} (Table 2: 29.3 and 12.8)")
print(f"mean of the per-dataset RER: {np.mean(rer[1:]):.1f}% (Table 2: 55.2%)")
print(f"RER of the two averages: {100 * (a.mean() - b.mean()) / a.mean():.1f}%")
