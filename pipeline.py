from langchain.embeddings import OpenAIEmbeddings
import faiss
import numpy as np
import openai

class OpenAIPipeline:

    def __init__(self,config):
        self.config = config
        openai.api_key = config.openai_api_key

    def __enter__(self):
        return self

    def __exit__(self,*exc):
        pass

    def search_in_context(self,query,context):
        embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
        embedding_vector = embeddings.embed_query(context)

        # Create FAISS index
        dimension = 1536  # Example dimension for OpenAI embeddings
        index = faiss.IndexFlatL2(dimension)

        # Add vectors to the index
        vectors = np.array([embedding_vector], dtype=np.float32)
        index.add(vectors)

        # Search for the nearest neighbor
        query_vector = embeddings.embed_query(query)
        self.distances, self.indices = index.search(np.array([query_vector], dtype=np.float32), k=1)

    def search_in_gpt4(self,query,context):
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=300
        )
        return response["choices"][0]["text"]