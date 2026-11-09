"""FP16 range and the underflow rule from Section 3.1 and Section 3.2."""
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)  # the overflow to inf is the point

info = np.finfo(np.float16)
print("max normal        :", float(info.max))
print("min normal  2^-14 :", float(info.tiny), "=", 2.0**-14)
print("min subnormal 2^-24:", 2.0**-24, "=", float(np.nextafter(np.float16(0), np.float16(1))))
for name, v in [("2^-24", 2.0**-24), ("1.5 * 2^-25", 1.5 * 2.0**-25), ("2^-25", 2.0**-25), ("1e-8", 1e-8), ("1e-5", 1e-5)]:
    print(f"float16({name:12s}) = {float(np.float16(v))!r}")
print("float16(65504)    =", float(np.float16(65504)))
print("float16(65520)    =", float(np.float16(65520)))
print("float16(70000)    =", float(np.float16(70000)))
