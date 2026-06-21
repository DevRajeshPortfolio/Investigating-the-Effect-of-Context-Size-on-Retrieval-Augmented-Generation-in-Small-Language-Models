from rank_bm25 import BM25Okapi


class BM25Retriever:

    def __init__(self, chunks):

        self.chunks = chunks

        tokenized_chunks = [
            chunk.split()
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(tokenized_chunks)

    def retrieve(self, query, top_k=1):

        tokenized_query = query.split()

        results = self.bm25.get_top_n(
            tokenized_query,
            self.chunks,
            n=top_k
        )

        return results