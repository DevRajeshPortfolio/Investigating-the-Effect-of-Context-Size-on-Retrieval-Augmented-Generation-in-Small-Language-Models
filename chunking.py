def chunk_text(text, chunk_size=64, overlap=16):

    words = text.split()

    chunks = []

    step = chunk_size - overlap

    for i in range(0, len(words), step):

        chunk = words[i:i + chunk_size]

        chunks.append(" ".join(chunk))

    return chunks