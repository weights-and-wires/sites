---
id: "bpe"
title: "Neural Machine Translation of Rare Words with Subword Units"
authors: "Sennrich et al."
year: 2015
venue: "ACL 2016"
arxiv: "1508.07909"
url: "https://arxiv.org/abs/1508.07909"
track: nlp-llm
tier: core
est_hours: 2
status: full
summary: "A translation model with a fixed word vocabulary can't read or write words it hasn't seen, and the usual fix, a back-off dictionary, only copies them. The paper segments every word into subword units instead, using byte-pair encoding: start from characters and repeatedly merge the most frequent adjacent pair of symbols. The learned merges apply to any new word, so a fixed-size table gives an open vocabulary, and the number of merges sets the trade-off between vocabulary size and sequence length."
related_course: "ml: tokenization"
prereqs: []
background: [string-processing, frequency-counting]
course_links:
  - title: "Tokenisation: bytes, byte-pair encoding, and what a tokeniser charges"
    url: "https://ml.ifkash.dev/lessons/054-tokenisation"
self_check:
  - q: 'The paper represents the word `low` as `l o w </w>`. Why does it add the end-of-word symbol, and how many adjacent pairs does that word contain?'
    a: 'The symbol lets the decoder restore word boundaries after translation, and it lets a merge tell the end of a word from the middle: `er</w>` is a suffix, while `er` inside `era` is not. The word has 4 symbols and so 3 pairs: `(l, o)`, `(o, w)`, and `(w, </w>)`.'
    see: string-processing
  - q: 'A dictionary maps `ab` to 9 and `cd` to 1. What is the count of the pair `(a, b)` if you count over the corpus, and what is it if you count over distinct word types?'
    a: 'Over the corpus the count is 9, because every occurrence of the word contributes. Over word types it is 1. The paper runs on a dictionary of word types weighted by frequency (Section 3.2), so Algorithm 1 adds the frequency and not 1.'
    see: frequency-counting
  - q: 'In the `get_stats` function of Algorithm 1, why does the paper count pairs only inside words and never across a space?'
    a: 'Section 3.2 skips pairs that cross word boundaries for efficiency. That lets the algorithm run on the dictionary of word types, with each word weighted by its frequency, and not on the running text. It also keeps every learned symbol a part of one word.'
    see: string-processing
---

## The problem

A neural machine translation (NMT) model of 2015 reads and writes words from a fixed list. Section 1 says the vocabulary is typically limited to 30,000 to 50,000 words, while translation is an open-vocabulary problem. Every other word becomes an `UNK` token, and the model can't translate it or produce it.

The standard repair was a back-off dictionary. The model emits `UNK`, and a word-alignment table looks up or copies the source word (Section 1). That repair assumes that source and target words correspond one to one. Consider the German compound *Abwasserbehandlungsanlage* ("sewage water treatment plant"), which the paper uses in Section 1. English needs four words for it, and a word-level model has to treat it as one unrelated symbol. Copying also fails when the alphabets differ, because names then need transliteration.

The paper's observation (Section 3) is that a translator can handle many unseen words from smaller known pieces. In an analysis of 100 rare German tokens, the authors found 56 compounds, 21 names, 6 loanwords, 5 cases of affixation, 1 number, and 1 computer-language identifier. So the paper asks for a representation in which any word is a sequence of pieces, the piece inventory has a fixed size, and the pieces are frequent enough to learn.

Section 4.2 reports that this works. On WMT 2015 English to German, unigram F1 on rare words rises from 36.8% for the back-off baseline (`WDict`) to 41.8% for joint BPE (`BPE-J90k`) in Table 2. On English to Russian, BPE-J90k beats `WDict` by 1.3 BLEU and 2.0 CHRF3 in the ensemble setting (Table 3 and Section 4.2).

## Key ideas

The paper has no numbered equations, so the page states each definition in words and cites Algorithm 1, Figure 1, or a section. Notation here is mine and not the paper's: $w$ is a word, $f(w)$ its frequency in the training dictionary, $V_0$ the initial symbol vocabulary of characters, and $k$ the number of merges. Every number comes from a script in `scripts/`. Run any script from the paper folder to reproduce its output. All scripts import `scripts/bpe_ref.py`, which follows the paper's Algorithm 1.

### Learning merges

**Intuition.** BPE began as a data compression method (Gage, 1994): find the most frequent pair of adjacent bytes and replace it with a byte that doesn't occur in the data. The paper runs the same loop on characters. Frequent pairs such as `e r` become one symbol, and then `er` can pair with `</w>` or with other symbols. Frequent words and common endings end up as single symbols, and rare words stay split.

**Math.** Section 3.2 starts with each word as a sequence of characters plus an end-of-word symbol. The paper writes the symbol as "$\cdot$" in the text and as `</w>` in Algorithm 1. This page uses `</w>`. For adjacent symbols $(a, b)$, the count over the dictionary is

$$
\operatorname{count}(a, b) = \sum_{w} f(w)\,\#_{(a,b)}(w)
$$

where $\#_{(a,b)}(w)$ is the number of positions in $w$ where $a$ is followed by $b$. This formula is my restatement of the `get_stats` function in Algorithm 1, and the paper doesn't number it. Each step replaces every occurrence of the most frequent pair $(a, b)$ with the new symbol $ab$ (Section 3.2). Pairs never cross word boundaries. The final vocabulary has $|V_0| + k$ symbols: the initial characters plus one symbol per merge, so $k$ is the only hyperparameter (Section 3.2).

**Tiny example, computed by `scripts/figure1_example.py`.** Take the dictionary from the caption of Figure 1, {`low`: 5, `lowest`: 2, `newer`: 6, `wider`: 3}. The counts and the first four merges are as follows:

| Step | Result |
|---|---|
| Top pair counts | `(e, r)`: 9, `(r, </w>)`: 9, `(w, e)`: 8, `(l, o)`: 7, `(o, w)`: 7 |
| Merge 1 | `e r` becomes `er` (count 9) |
| Merge 2 | `er </w>` becomes `er</w>` (count 9) |
| Merge 3 | `l o` becomes `lo` (count 7) |
| Merge 4 | `lo w` becomes `low` (count 7) |

Two pairs tie at 9 in the first step. Algorithm 1 uses `max(pairs, key=pairs.get)`, which returns the pair that the dictionary saw first, so the result depends on insertion order. This page and the exercises break ties by taking the lexicographically smallest pair, so the result is the same on every machine. The first four merges are the same either way.

The paper's Figure 1 is inconsistent with its own listing, and you should know that before you compare outputs. The caption names the dictionary {`low`, `lowest`, `newer`, `wider`}, while the listing in Algorithm 1 uses `lower`, `newest`, and `widest`. The merges printed in Figure 1 (`r ·`, `l o`, `lo w`, `e r·`) also don't come out of either dictionary in that order. The script shows that the caption's dictionary gives `e r` first and then `er </w>`. Treat Figure 1 as an illustration, not as a test case.

Exercise 1 asks you to write the pair counter and the merge step. Exercise 2 asks you to write the full learner.

### Encoding new words and the open vocabulary

**Intuition.** A learned merge list is a recipe. To segment any word, start from its characters and replay the recipe in the order it was learned. Common words collapse to one symbol, a rare word stops partway, and a word of unseen letters stays as characters.

**Math.** At test time, Section 3.2 splits a word into characters and then applies the learned operations to merge them into larger, known symbols. The order matters, because each merge was the best choice given the earlier ones, so earlier merges have priority. The method is open-vocabulary because every word is at worst a string of known characters. Footnote 3 lists the two cases in which a symbol is unknown at test time: an unknown character, and a symbol whose every training occurrence was merged into a larger symbol (the paper's example is `safeguar`).

**Tiny example, computed by `scripts/figure1_example.py`.** With the ten merges learned from the dictionary above:

| Word | Segmentation |
|---|---|
| `lower` (not in training) | `low er</w>` |
| `lowest` | `low e s t </w>` |
| `newer` | `newer</w>` |
| `wider` | `w ider</w>` |
| `slow` (not in training) | `s low</w>` |
| `xyz` | `x y z </w>` |

The paper makes the same point about `lower` in Section 3.2: the unseen word is segmented into `low er·`. The tool doesn't care that `lower` was never in the training dictionary. Notice two things. `lowest` was in the training dictionary and still comes out as `low e s t`, because no merge for `est` ranked in the top ten. And `ider</w>` exists only because `wider` is frequent enough in this dictionary and shares `er</w>` with `newer`. The piece is a statistical artifact with no meaning of its own.

Exercise 2 asks you to write `encode`.

### The vocabulary-size trade-off

**Intuition.** More merges mean longer symbols, so the text takes fewer tokens, and each token has been seen more rarely. Fewer merges mean a small vocabulary and long sequences. Section 3.1 says the goal is a representation that is compact in both ways: a small vocabulary saves time and memory, but a longer text reduces efficiency and increases the distance over which a neural model must pass information.

**Math.** With $k$ merges the symbol vocabulary has size $|V_0| + k$, where $V_0$ includes the end-of-word symbol in this page's accounting (Section 3.2). The number of tokens falls as $k$ grows. The compression ratio on this page is

$$
\text{ratio}(k) = \frac{\sum_w f(w)\,(|w| + 1)}{\sum_w f(w)\,|\operatorname{encode}_k(w)|}
$$

The numerator is the length with zero merges. The paper doesn't define a ratio. It reports token counts and types in Table 1.

**Tiny example, computed by `scripts/figure1_example.py` and `scripts/compression.py`.** On the four-word dictionary, zero merges give 88 symbols and a vocabulary of 11. Four merges give 56 tokens and a vocabulary of 15. Ten merges give 27 tokens and a vocabulary of 21, which is a compression of 3.26.

The second script uses a synthetic corpus of 2,460 word types and 166,501 word tokens, built from stems, prefixes, and suffixes with a Zipf-like frequency rule in `scripts/toy_corpus.py`. It isn't real language, so read the shape and not the values:

| Merges | Vocabulary | Tokens | Tokens per word | Compression |
|---|---|---|---|---|
| 0 | 26 | 1,566,033 | 9.41 | 1.00 |
| 100 | 126 | 488,143 | 2.93 | 3.21 |
| 500 | 526 | 232,632 | 1.40 | 6.73 |
| 2000 | 2,026 | 175,126 | 1.05 | 8.94 |

The first 100 merges cut the length by two thirds. The next 1,900 buy much less. By 2,000 merges the text is close to one token per word, so the vocabulary is nearly word-level. The script also segments `rewatchingly`, which is not in the corpus. With 100 merges it becomes `rew atch in g ly</w>`, and with 2,000 it becomes `rewatch in g ly</w>`.

Table 1 of the paper shows the same shape on the German training data (about 100 million words). Characters give 550 million tokens with 3,000 types. BPE with 59,500 merges gives 112 million tokens with 63,000 types, and the unsegmented text has 1,750,000 types. In the paper's numbers BPE costs only 12% more tokens than words, and it has no unknown symbols in the development set (Table 1, `# UNK`: 0, against 1,079 for the word-level text).

Exercise 3 asks you to compute these statistics.

<!-- exercises -->

## Pitfalls

- **Merging text in place with `str.replace`.** The pair `(b, c)` is written `b c`. The string `b c` also appears inside `ab c`, where `ab` is one symbol, so a plain replace turns `ab c` into the wrong `abc`. The paper's `merge_vocab` uses a regular expression with `(?<!\S)` and `(?!\S)` to require whole symbols. `scripts/figure1_example.py` shows the failure.
- **Counting types and not tokens.** The dictionary stores each word once with its frequency. Add the frequency when you count a pair. If you add 1, rare words are as influential as common words, and the merges stop reflecting the corpus.
- **Forgetting the end-of-word symbol.** Without it, `er` at the end of `newer` and `er` in the middle of `era` are the same symbol, and you can't tell where words end after decoding.
- **Applying the merges in the wrong order.** The list is ordered by priority. Sorting it, reversing it, or applying only the first rule that matches gives different segmentations from the training run.
- **Relying on dictionary order for ties.** `max` over a dict returns the first maximal pair, so two runs with a different word order can learn different merges. Fix a tie-break rule.
- **Reading the vocabulary size as the number of merges.** The size is the number of merges plus the initial characters. Real byte-level tokenizers also add special tokens.
- **Learning separate source and target encodings.** Section 3.2 compares independent encodings with joint BPE, learned on the union of both vocabularies. Independent encodings are more compact, but the same name may be segmented differently on the two sides. In Table 2, joint BPE (`BPE-J90k`) scores 41.8 unigram F1 on rare words against 40.9 for `BPE-60k`, and the paper concludes that joint learning is more effective (Section 4.2).
- **Trusting BLEU alone.** Rare and unseen words are only 9 to 11 percent of the test sets (Section 4.2). The paper reports unigram F1 on rare and out-of-vocabulary words for that reason.
- **Assuming this is the tokenizer in a modern LLM.** Later models, such as GPT-2, apply the same merge idea to bytes. The paper works on characters (Section 3.2), so it needs a handler for unknown characters (footnote 3), which byte-level BPE doesn't.

## Further reading

- [subword-nmt](https://github.com/rsennrich/subword-nmt) is the authors' reference implementation of the learner and the encoder.
- [minbpe](https://github.com/karpathy/minbpe), by Andrej Karpathy, is a short, readable byte-level BPE trainer and tokenizer.
- Philip Gage, "A New Algorithm for Data Compression", *The C Users Journal*, 1994, is the compression method that the paper adapts (Section 3.2).
- [Tokenisation: bytes, byte-pair encoding, and what a tokeniser charges](https://ml.ifkash.dev/lessons/054-tokenisation) covers the byte-level variant and the cost of a tokenizer in more depth.
