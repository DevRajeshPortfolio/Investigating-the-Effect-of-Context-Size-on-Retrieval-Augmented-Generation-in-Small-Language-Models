NON_RAG_TEMPLATE = """
Answer the following question briefly and factually.

Question:
{question}

Answer:
"""

RAG_TEMPLATE = """
Use the provided context to answer the question factually.

Context:
{context}

Question:
{question}

Answer:
"""