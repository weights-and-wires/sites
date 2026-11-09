"""Check the parameter counts in Section 3.4 from the stated configuration."""

LAYERS, CELLS, EMB = 4, 1000, 1000
SRC_VOCAB, TGT_VOCAB = 160_000, 80_000

# One LSTM layer has four weight blocks (three gates and the cell input) that
# each read the layer input (1000) and the previous state (1000). Biases are ignored.
per_layer = 4 * (CELLS * CELLS + CELLS * CELLS)
recurrent_enc = LAYERS * per_layer
recurrent_dec = LAYERS * per_layer
src_emb = SRC_VOCAB * EMB
tgt_emb = TGT_VOCAB * EMB
softmax = TGT_VOCAB * CELLS
total = recurrent_enc + recurrent_dec + src_emb + tgt_emb + softmax

if __name__ == "__main__":
    print(f"per layer         {per_layer / 1e6:6.1f}M")
    print(f"encoder LSTM      {recurrent_enc / 1e6:6.1f}M")
    print(f"decoder LSTM      {recurrent_dec / 1e6:6.1f}M")
    print(f"source embedding  {src_emb / 1e6:6.1f}M")
    print(f"target embedding  {tgt_emb / 1e6:6.1f}M")
    print(f"output softmax    {softmax / 1e6:6.1f}M")
    print(f"total             {total / 1e6:6.1f}M")
    print(f"state size per sentence: {2 * LAYERS * CELLS} numbers (h and c over {LAYERS} layers)")
