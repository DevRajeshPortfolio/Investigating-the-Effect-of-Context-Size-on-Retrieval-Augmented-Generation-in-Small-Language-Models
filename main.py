from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from data.load_data import load_triviaqa_sample
from retrieval.chunking import chunk_text
from retrieval.bm25_retriever import BM25Retriever


MODEL_NAME = "microsoft/phi-2"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Model loaded.\n")


# ============================================
# LOAD DATA
# ============================================

sample = load_triviaqa_sample()

question = sample["question"]
ground_truth = sample["answer"]
context = sample["context"]

print("QUESTION:")
print(question)

print("\nGROUND TRUTH:")
print(ground_truth)



chunks = chunk_text(
    context,
    chunk_size=80,
    overlap=20
)

print(f"\nTotal chunks created: {len(chunks)}")



retriever = BM25Retriever(chunks)

top_chunks = retriever.retrieve(
    question,
    top_k=3
)

print("\nTOP RETRIEVED CHUNKS:\n")

for i, chunk in enumerate(top_chunks):
    print(f"CHUNK {i+1}:")
    print(chunk)
    print()




retrieved_context = "\n".join(top_chunks)

prompt = f"""
Answer the question using the context below.

Context:
{retrieved_context}

Question:
{question}

Answer:
"""


print("\nFINAL PROMPT:\n")
print(prompt[:1000])





tokens = tokenizer.encode(prompt)

print(f"\nPROMPT TOKEN COUNT: {len(tokens)}")



inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=30,
        temperature=0.0
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nMODEL OUTPUT:\n")
print(response)