def build_prompt(language, task, timestamps, prev_tokens=None):
    """Return the decoder prompt as a list of token strings.

    language: a code such as "en" or "es". task: "transcribe" or "translate".
    timestamps: True to let the model predict timestamp tokens.
    prev_tokens: optional list of text tokens that precede this 30 s window.
    """
    if task not in ("transcribe", "translate"):
        raise ValueError(f"task must be 'transcribe' or 'translate', got {task!r}")
    out = []
    if prev_tokens:
        out += ["<|startofprev|>"] + list(prev_tokens)
    out += ["<|startoftranscript|>", f"<|{language}|>", f"<|{task}|>"]
    if not timestamps:
        out.append("<|notimestamps|>")
    return out


def prompt_loss_mask(tokens):
    """Return a list of bools, one per token: False for previous-text tokens, True otherwise."""
    mask = [True] * len(tokens)
    if "<|startofprev|>" in tokens:
        a = tokens.index("<|startofprev|>")
        b = tokens.index("<|startoftranscript|>")
        for i in range(a + 1, b):
            mask[i] = False
    return mask
