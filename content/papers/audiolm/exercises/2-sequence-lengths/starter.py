import math


def sequence_lengths(seconds, sr=16000, acoustic_hop=320, semantic_hop=640, Q=12, Qp=4, chunk_s=3):
    """Return a dict of token counts for a clip of `seconds` seconds.

    Keys: semantic, acoustic_coarse, acoustic_all, stage1, stage2,
    stage3_chunk, stage3_chunks, one_stage.
    """
    raise NotImplementedError
