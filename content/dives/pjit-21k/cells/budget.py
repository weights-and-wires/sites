# Part 3: the training bill, from the numbers the post states.
steps_512 = 610.27e6 / 2048
steps_1024 = 49.97e6 / 2048
print(f"steps at batch 2,048: 512 stage {steps_512:,.0f} (post: 297,985), 1024 stage {steps_1024:,.0f} (post: 24,400)")

pre_hours, post_hours, price = 8139, 276, 2.50
print(f"pretraining   {pre_hours:,} h x ${price} = ${pre_hours * price:,.1f}")
print(f"post-training {post_hours:,} h x ${price} = ${post_hours * price:,.1f}")
total_h = pre_hours + post_hours
print(f"total         {total_h:,} h x ${price} = ${total_h * price:,.1f}")

samples = 610.27e6 + 49.97e6
print(f"\nsamples per pretraining H100-hour: {samples / pre_hours:,.0f}")
print(f"hours per 1M samples:              {pre_hours / (samples / 1e6):.1f}")
# Assumes all 64 GPUs are busy the whole time. The post's hours include restarts,
# so the real wall-clock time was longer than this.
print(f"64 GPUs, no downtime: {pre_hours / 64:.0f} h = {pre_hours / 64 / 24:.1f} days")
print(f"images per GPU-second at that rate: {samples / (pre_hours * 3600):.1f}")
result = {"hours": pre_hours, "cost": total_h * price}
