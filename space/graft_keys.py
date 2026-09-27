"""Map merged text weights to their matching vision-language parent."""


def text_to_parent(key: str) -> str:
    if key == "lm_head.weight" or key.startswith("model.language_model."):
        return key
    if key.startswith("model."):
        return "model.language_model." + key[len("model."):]
    raise ValueError(f"unexpected key in the text checkpoint: {key}")
