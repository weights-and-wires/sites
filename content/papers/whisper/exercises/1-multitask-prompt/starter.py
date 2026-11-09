def build_prompt(language, task, timestamps, prev_tokens=None):
    """Return the decoder prompt as a list of token strings.

    language: a code such as "en" or "es". task: "transcribe" or "translate".
    timestamps: True to let the model predict timestamp tokens.
    prev_tokens: optional list of text tokens that precede this 30 s window.
    """
    raise NotImplementedError


def prompt_loss_mask(tokens):
    """Return a list of bools, one per token: False for previous-text tokens, True otherwise."""
    raise NotImplementedError
