#!/usr/bin/env python3
"""How little feedback the Atari runs use (Section 3.1.2, Appendix A.2, Abstract)."""

labels = 5500          # queries to the human on Atari (Section 3.1.2)
steps_per_segment = 25  # timesteps per segment (Appendix A.2)
rl_timesteps = 50e6     # timesteps that the runs actually trained for (Appendix A.2)
seconds_per_query = (3, 5)  # contractor response time (Section 3.1)

shown = labels * 2 * steps_per_segment
print(f"labels: {labels}; each shows 2 segments of {steps_per_segment} timesteps = {shown:,} timesteps shown")
print(f"fraction of the {rl_timesteps:,.0f} RL timesteps shown to a human: {shown / rl_timesteps:.2%}")
print(f"labels per RL timestep: 1 in {rl_timesteps / labels:,.0f}")
lo, hi = (labels * s / 3600 for s in seconds_per_query)
print(f"human time at {seconds_per_query[0]}-{seconds_per_query[1]} s per query: {lo:.1f} to {hi:.1f} hours")
