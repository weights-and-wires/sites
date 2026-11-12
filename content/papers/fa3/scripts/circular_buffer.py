"""Producer and consumer on a circular shared-memory buffer (Section 3.1,
Algorithm 1, lines 6 to 10 for the producer and 15 to 22 for the consumer).

The producer loads K_j and V_j into slot j % s. It must wait for the consumer
to release that slot, but only after the buffer has filled once.

Run: python3 scripts/circular_buffer.py
"""

s, Tc = 2, 6
load, compute = 1.0, 3.0   # toy times for one (K_j, V_j) load and one consumer iteration

slot_free = [0.0] * s       # time at which the consumer releases each slot
producer_t = 0.0
consumer_t = 0.0
print(f"s = {s} stages, Tc = {Tc} blocks, load = {load}, compute = {compute}")
print("j  slot  producer waits?  load done  consumer starts  consumer done")
for j in range(Tc):
    slot = j % s
    waits = j >= s
    start = max(producer_t, slot_free[slot]) if waits else producer_t
    loaded = start + load
    producer_t = loaded
    c_start = max(consumer_t, loaded)
    consumer_t = c_start + compute
    slot_free[slot] = consumer_t
    print(f"{j}  {slot:<4}  {str(waits):<15}  {loaded:9.1f}  {c_start:15.1f}  {consumer_t:13.1f}")
print(f"The consumer is busy {Tc * compute} of {consumer_t} time units, so its only idle time is the first load: {consumer_t - Tc * compute}")
