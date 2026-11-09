from submission import basic_block_params, bottleneck_params, conv_params, projection_shortcut_extra_params


def test_conv_params():
    assert conv_params(3, 64, 64) == 36864, f"expected 3*3*64*64 = 36864, got {conv_params(3, 64, 64)}"
    assert conv_params(1, 256, 64) == 16384, f"expected 1*1*256*64 = 16384, got {conv_params(1, 256, 64)}"
    assert conv_params(3, 16, 32) == 4608, f"expected 3*3*16*32 = 4608 (c_in and c_out differ), got {conv_params(3, 16, 32)}"


def test_basic_block():
    assert basic_block_params(64) == 73728, f"expected 2 * 36864 = 73728, got {basic_block_params(64)}"
    assert basic_block_params(512) == 4718592, f"expected 2 * 3*3*512*512 = 4718592, got {basic_block_params(512)}"


def test_bottleneck_matches_figure_5():
    got = bottleneck_params(256, 64)
    assert got == 69632, f"expected 16384 + 36864 + 16384 = 69632, got {got}"


def test_bottleneck_is_cheaper_than_basic_at_the_same_ends():
    # a 256-wide basic block, against a bottleneck with 256 ends and a 64-wide middle
    basic, bott = basic_block_params(256), bottleneck_params(256, 64)
    assert basic == 1179648, f"expected 2 * 3*3*256*256 = 1179648, got {basic}"
    assert bott < basic / 10, f"expected the bottleneck ({bott}) to be under a tenth of {basic}"


def test_other_stages():
    assert bottleneck_params(2048, 512) == 2 * 2048 * 512 + 9 * 512 * 512, f"expected {2 * 2048 * 512 + 9 * 512 * 512}, got {bottleneck_params(2048, 512)}"
    assert bottleneck_params(512, 128) == 2 * 512 * 128 + 9 * 128 * 128, f"expected {2 * 512 * 128 + 9 * 128 * 128}, got {bottleneck_params(512, 128)}"


def test_projection_shortcut_nearly_doubles_the_block():
    extra = projection_shortcut_extra_params(256)
    assert extra == 65536, f"expected 1*1*256*256 = 65536, got {extra}"
    ratio = (bottleneck_params(256, 64) + extra) / bottleneck_params(256, 64)
    assert 1.9 < ratio < 2.0, f"the paper says a projection roughly doubles the bottleneck; expected a ratio near 1.94, got {ratio:.3f}"
