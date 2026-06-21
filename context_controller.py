def truncate_prompt(prompt, tokenizer, max_tokens):

    tokens = tokenizer.encode(
        prompt,
        truncation=True,
        max_length=max_tokens
    )

    truncated_prompt = tokenizer.decode(tokens)

    return truncated_prompt