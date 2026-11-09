#!/usr/bin/env python3
"""Decoder token sequences for the multitask format (Section 2.3, Figure 1)."""

from whisper_ref import build_prompt, build_target, timestamp_token

print("transcribe, no timestamps:")
print(" ", build_prompt("en", "transcribe", False))
print("translate Spanish audio to English, timestamps:")
print(" ", build_prompt("es", "translate", True))
print("with previous text:")
print(" ", build_prompt("en", "transcribe", False, prev_tokens=["Ask", " not"]))

print("\ntimestamp tokens (20 ms grid):")
for s in (0.0, 0.009, 0.031, 2.4, 29.98, 30.0):
    print(f"  {s:6.3f} s -> {timestamp_token(s)}")
print(f"tokens for 30 s: {int(30 / 0.02) + 1} (0.00 to 30.00 inclusive)")

segments = [(0.0, 2.4, [" Ask", " not"]), (2.4, 5.0, [" what", " your", " country"])]
target = build_target(segments, True)
print("\ntarget with timestamps:")
print(" ", target)
print("target without:")
print(" ", build_target(segments, False))
full = build_prompt("en", "transcribe", True) + target
print(f"\nfull sequence length with timestamps: {len(full)}")
