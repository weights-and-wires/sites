"""AR and NAR attention masks for a tiny VALL-E sequence (Figure 3, Section 4.2.1-4.2.2)."""
import numpy as np

n_text, n_audio = 3, 3  # text holds two phonemes plus <EOS>
n = n_text + n_audio


def ar_mask(n_text, n_audio):
    n = n_text + n_audio
    m = np.zeros((n, n), dtype=bool)
    m[:n_text, :n_text] = True
    m[n_text:, :n_text] = True
    m[n_text:, n_text:] = np.tril(np.ones((n_audio, n_audio), dtype=bool))
    return m


print("AR mask (rows = queries, columns = keys):")
print(ar_mask(n_text, n_audio).astype(int))
print("NAR mask: all", n * n, "entries are 1")
print("AR keys visible to audio position 0:", int(ar_mask(n_text, n_audio)[n_text].sum()))
print("AR keys visible to audio position 2:", int(ar_mask(n_text, n_audio)[n_text + 2].sum()))
