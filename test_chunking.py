from data.load_data import load_triviaqa_sample
from retrieval.chunking import chunk_text


data = load_triviaqa_sample(1)

sample = data[0]

context = sample["context"]

chunks = chunk_text(
    context,
    chunk_size=64,
    overlap=16
)

print(f"\nTOTAL CHUNKS: {len(chunks)}")

print("\nFIRST CHUNK:\n")

print(chunks[0])

print("\nSECOND CHUNK:\n")

print(chunks[1])