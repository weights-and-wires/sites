import math


def sequence_lengths(seconds, sr=16000, acoustic_hop=320, semantic_hop=640, Q=12, Qp=4, chunk_s=3):
    """Return a dict of token counts for a clip of `seconds` seconds.

    Keys: semantic, acoustic_coarse, acoustic_all, stage1, stage2,
    stage3_chunk, stage3_chunks, one_stage.
    """
    T = int(seconds * sr)
    TS = T // semantic_hop
    TA = T // acoustic_hop
    chunk_TA = int(chunk_s * sr) // acoustic_hop
    return {
        "semantic": TS,
        "acoustic_coarse": TA * Qp,
        "acoustic_all": TA * Q,
        "stage1": TS,
        "stage2": TS + TA * Qp,
        "stage3_chunk": chunk_TA * Q,
        "stage3_chunks": math.ceil(seconds / chunk_s),
        "one_stage": TS + TA * Q,
    }
