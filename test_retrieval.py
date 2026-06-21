from data.load_data import load_triviaqa_sample
from retrieval.chunking import chunk_text
from retrieval.bm25_retriever import BM25Retriever


data = load_triviaqa_sample(1)

sample = data[0]

question = sample["question"]

context = sample["context"]

chunks = chunk_text(
    context,
    chunk_size=64,
    overlap=16
)

retriever = BM25Retriever(chunks)

results = retriever.retrieve(
    question,
    top_k=3
)

print("\nQUESTION:\n")
print(question)

print("\nTOP RETRIEVED CHUNKS:\n")

for i, result in enumerate(results):

    print(f"\nRESULT {i+1}:\n")

    print(result)