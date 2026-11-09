from submission import conv_output_length, receptive_field

KERNELS = (10, 3, 3, 3, 3, 2, 2)
STRIDES = (5, 2, 2, 2, 2, 2, 2)


def test_paper_encoder_receptive_field_and_stride():
    rf, stride = receptive_field(KERNELS, STRIDES)
    assert stride == 320, f"total stride: expected 320 samples (20 ms at 16 kHz), got {stride}"
    assert rf == 400, f"receptive field: expected 400 samples (25 ms), got {rf}"


def test_receptive_field_small_cases():
    assert receptive_field((3,), (1,)) == (3, 1), f"one k=3 s=1 layer: expected (3, 1), got {receptive_field((3,), (1,))}"
    assert receptive_field((3, 3), (1, 1)) == (5, 1), f"two k=3 s=1 layers: expected (5, 1), got {receptive_field((3, 3), (1, 1))}"
    # The second layer sees 2 outputs of the first, which are 4 samples apart.
    assert receptive_field((4, 2), (4, 1)) == (8, 4), f"k=(4,2) s=(4,1): expected (8, 4), got {receptive_field((4, 2), (4, 1))}"
    # Stride before kernel matters: the order of layers changes the receptive field.
    a = receptive_field((2, 5), (2, 1))
    b = receptive_field((5, 2), (1, 2))
    assert a == (10, 2) and b == (6, 2), f"order matters: expected (10, 2) and (6, 2), got {a} and {b}"


def test_one_second_of_audio_gives_49_frames():
    n = conv_output_length(16000, KERNELS, STRIDES)
    assert n == 49, f"1 s at 16 kHz: expected 49 frames, got {n}"


def test_base_crop_gives_781_frames():
    n = conv_output_length(250_000, KERNELS, STRIDES)
    assert n == 781, f"250k samples: expected 781 frames, got {n}"


def test_layer_by_layer_not_one_division():
    # For 16000 samples, 16000 // 320 = 50, but the edge losses leave 49.
    n = conv_output_length(16000, KERNELS, STRIDES)
    assert n != 16000 // 320, f"expected 49, which differs from the naive 16000 // 320 = 50, got {n}"


def test_inputs_shorter_than_the_receptive_field():
    assert conv_output_length(399, KERNELS, STRIDES) == 0, f"399 samples: expected 0 frames, got {conv_output_length(399, KERNELS, STRIDES)}"
    assert conv_output_length(400, KERNELS, STRIDES) == 1, f"400 samples: expected 1 frame, got {conv_output_length(400, KERNELS, STRIDES)}"
    assert conv_output_length(5, (10,), (5,)) == 0, f"n < k: expected 0, got {conv_output_length(5, (10,), (5,))}"


def test_one_frame_per_stride_beyond_the_receptive_field():
    a = conv_output_length(400, KERNELS, STRIDES)
    b = conv_output_length(400 + 320, KERNELS, STRIDES)
    assert b == a + 1, f"320 more samples add one frame: expected {a + 1}, got {b}"


def test_too_short_inputs_give_zero_not_negative():
    got = conv_output_length(3, (10,), (5,))
    assert got == 0, f"3 samples, k = 10: expected 0 frames, got {got}"
    got = conv_output_length(11, (10, 5), (5, 2))
    assert got == 0, f"11 samples, then 1 frame is shorter than k = 5: expected 0, got {got}"
