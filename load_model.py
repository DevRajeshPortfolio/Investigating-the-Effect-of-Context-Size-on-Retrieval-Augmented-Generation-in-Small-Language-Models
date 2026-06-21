from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

import torch


MODEL_NAME = "microsoft/phi-2"


def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    return tokenizer, model