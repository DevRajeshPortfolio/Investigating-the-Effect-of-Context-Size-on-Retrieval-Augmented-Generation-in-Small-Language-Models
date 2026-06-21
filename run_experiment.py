from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
from tqdm import tqdm

from data.load_data import load_triviaqa_dataset
from retrieval.chunking import chunk_text
from retrieval.bm25_retriever import BM25Retriever


MODEL_NAME = "microsoft/phi-2"

NUM_EXAMPLES = 20

TOP_K = 3

MAX_CONTEXT_TOKENS = 128


print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Model loaded.\n")


dataset = load_triviaqa_dataset()

dataset = dataset.select(range(NUM_EXAMPLES))


def truncate_to_token_limit(text, max_tokens):

    tokens = tokenizer.encode(text)

    tokens = tokens[:max_tokens]

    return tokenizer.decode(tokens)


results = []

for sample in tqdm(dataset):

    question = sample["question"]

    ground_truth = sample["answer"]["value"]

    search_contexts = sample["search_results"]["search_context"]

    # Skip bad examples
    if len(search_contexts) == 0:
        continue

    context = search_contexts[0]

    chunks = chunk_text(
        context,
        chunk_size=80,
        overlap=20
    )

    retriever = BM25Retriever(chunks)

    top_chunks = retriever.retrieve(
        question,
        top_k=TOP_K
    )

    print("\nQUESTION:")
    print(question)

    print("\nTOP CHUNK:")
    print(top_chunks[0][:500])

    print("\n" + "=" * 80)

    retrieved_context = "\n".join(top_chunks[:2])

    retrieved_context = truncate_to_token_limit(
        retrieved_context,
        MAX_CONTEXT_TOKENS
    )

    prompt = f"""
Question:
{question}

Short Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    prediction = decoded.split("Short Answer:")[-1].strip()

    print("\nPREDICTION:")
    print(prediction)

    print("\nGROUND TRUTH:")
    print(ground_truth)

    print("\n" + "=" * 80)

    results.append({
        "question": question,
        "ground_truth": ground_truth,
        "model_output": prediction
    })


df = pd.DataFrame(results)

df.to_csv(
    "results_no_rag.csv",
    index=False
)

print("\nResults saved.")