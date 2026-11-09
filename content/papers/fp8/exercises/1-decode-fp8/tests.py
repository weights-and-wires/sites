import math
from submission import decode


def test_e4m3_table_1_extremes():
    cases = {
        0b01111110: 448.0,          # max normal, 1.75 * 2^8
        0b00001000: 2.0**-6,        # min normal
        0b00000111: 0.875 * 2.0**-6,  # max subnormal
        0b00000001: 2.0**-9,        # min subnormal
    }
    for code, want in cases.items():
        got = decode(code, "e4m3")
        assert got == want, f"E4M3 code {code:08b}: expected {want}, got {got}"


def test_e5m2_table_1_extremes():
    cases = {
        0b01111011: 57344.0,        # max normal, 1.75 * 2^15
        0b00000100: 2.0**-14,       # min normal
        0b00000011: 0.75 * 2.0**-14,  # max subnormal
        0b00000001: 2.0**-16,       # min subnormal
    }
    for code, want in cases.items():
        got = decode(code, "e5m2")
        assert got == want, f"E5M2 code {code:08b}: expected {want}, got {got}"


def test_e4m3_reclaimed_top_binade():
    want = [256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0]
    got = [decode(0b01111000 | m, "e4m3") for m in range(7)]
    assert got == want, f"exponent field 1111 should give {want}, got {got}"
    assert math.isnan(decode(0b01111111, "e4m3")), f"S.1111.111 must be NaN, got {decode(0b01111111, 'e4m3')}"
    assert math.isnan(decode(0b11111111, "e4m3")), "the negative NaN pattern must be NaN too"


def test_e4m3_has_no_infinity():
    infs = [c for c in range(256) if math.isinf(decode(c, "e4m3"))]
    assert infs == [], f"E4M3 has no infinities, but codes {infs} decode to inf"


def test_e5m2_specials():
    assert decode(0b01111100, "e5m2") == math.inf, f"expected +inf, got {decode(0b01111100, 'e5m2')}"
    assert decode(0b11111100, "e5m2") == -math.inf, f"expected -inf, got {decode(0b11111100, 'e5m2')}"
    for code in (0b01111101, 0b01111110, 0b01111111):
        assert math.isnan(decode(code, "e5m2")), f"code {code:08b} should be NaN, got {decode(code, 'e5m2')}"


def test_sign_and_zero():
    assert decode(0b10111000, "e4m3") == -1.0, f"expected -1.0, got {decode(0b10111000, 'e4m3')}"
    assert decode(0b10111100, "e5m2") == -1.0, f"expected -1.0, got {decode(0b10111100, 'e5m2')}"
    assert decode(0, "e4m3") == 0.0 and decode(0, "e5m2") == 0.0, "code 0 must decode to 0.0"
    assert math.copysign(1, decode(0b10000000, "e4m3")) == -1, "code 0x80 must decode to -0.0"


def test_value_counts():
    finite4 = sum(math.isfinite(decode(c, "e4m3")) for c in range(256))
    finite5 = sum(math.isfinite(decode(c, "e5m2")) for c in range(256))
    assert finite4 == 254, f"E4M3 should have 254 finite codes (256 minus 2 NaN), got {finite4}"
    assert finite5 == 248, f"E5M2 should have 248 finite codes (256 minus 2 inf and 6 NaN), got {finite5}"
