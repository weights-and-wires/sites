#!/usr/bin/env python3
"""Numbers for the Opteron X2 and X4 examples in Sections 3 and 4 of the paper.

Run from anywhere: python3 scripts/roofline_example.py
"""


def attainable(oi, peak, bw):
    """Attainable GFLOP/s = min(peak GFLOP/s, bandwidth GB/s x intensity)."""
    return min(peak, bw * oi)


# Section 3: Opteron X2 (peak 17.6 GFLOP/s, 15 GB/s measured bandwidth).
x2_peak, x2_bw = 17.6, 15.0
# Section 3 and Table 1: the X4 has "slightly more than four times" the X2 peak
# (2 sockets x 4 cores x 4 FLOPs/cycle x 2.3 GHz) and the Table 1 Stream bandwidth.
x4_peak = 2 * 4 * 4 * 2.3
x4_bw = 16.6

print("Opteron X2 roofline (peak %.1f GFLOP/s, %.1f GB/s)" % (x2_peak, x2_bw))
print("  ridge point: %.3f FLOPs/byte" % (x2_peak / x2_bw))
for oi in (0.25, 0.5, 1.0, 2.0, 4.0):
    p = attainable(oi, x2_peak, x2_bw)
    regime = "memory-bound" if oi * x2_bw < x2_peak else "compute-bound"
    print("  intensity %-4s -> %5.2f GFLOP/s (%s)" % (oi, p, regime))

print("  SpMV at intensity 0.25 reaches %.1f%% of peak" % (100 * attainable(0.25, x2_peak, x2_bw) / x2_peak))
print("Opteron X4 roofline (peak %.1f GFLOP/s, %.1f GB/s)" % (x4_peak, x4_bw))
print("  peak ratio X4 / X2: %.2f" % (x4_peak / x2_peak))
print("  ridge point: %.2f FLOPs/byte" % (x4_peak / x4_bw))
print("  intensity 1.0 on X4: %.2f GFLOP/s (X2: %.2f)" % (attainable(1.0, x4_peak, x4_bw), attainable(1.0, x2_peak, x2_bw)))
print("  intensity 4.0 on X4: %.2f GFLOP/s (X2: %.2f)" % (attainable(4.0, x4_peak, x4_bw), attainable(4.0, x2_peak, x2_bw)))

print("Opteron X2 ceilings (Section 4)")
for oi in (0.5, 1.0):
    for name, c_peak, c_bw in (
        ("roofline", 17.6, 15.0),
        ("no software prefetch", 17.6, 11.0),
        ("no memory affinity", 17.6, 4.8),
        ("unit stride only", 17.6, 2.7),
    ):
        print("  intensity %-3s %-22s -> %5.2f GFLOP/s" % (oi, name, attainable(oi, c_peak, c_bw)))
print("  intensity 4.0 with ILP/SIMD missing (2.2 GFLOP/s ceiling): %.2f GFLOP/s" % attainable(4.0, 2.2, x2_bw))

print("Intensity of an achieved result (Table 4, SpMV on the Opteron X4)")
gflops, oi = 4.2, 0.25
print("  %.1f GFLOP/s at %.2f FLOPs/byte -> %.1f GB/s of DRAM traffic" % (gflops, oi, gflops / oi))
print("  Table 4 lists a Stream bandwidth ceiling of 17.6 GB/s, so the kernel reaches %.0f%% of it" % (100 * (gflops / oi) / 17.6))

print("Operational intensity by counting (Sections 6.3.3 and 7)")
print("  7-point stencil: 8 FLOPs / 24 bytes = %.3f" % (8 / 24))
print("  3-D FFT transpose exchange: 1 exchange / 32 bytes = %.5f (= 1/32 = %.5f)" % (1 / 32, 1 / 32))
