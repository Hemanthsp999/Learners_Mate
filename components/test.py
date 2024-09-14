import FaissBase
from sentence_transformers import SentenceTransformer

# Configuration
FAISS_INDEX_FILE = "/home/hexa/LearnersMate/faiss_index_file.index"
MODEL_NAME = 'all-mpnet-base-v2'
TOP_K_RESULTS = 5

# Load the model and FAISS index
print(f"Loading the SentenceTransformer model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

print(f"Loading the FAISS index from: {FAISS_INDEX_FILE}")
index = FaissBase.Initialize()

'''
if not index.is_trained:
    print("Model is not trained")
else:
    print("Model is trained")
'''


def encode_query(query):
    """Encode the query using the SentenceTransformer model."""
    return model.encode([query], convert_to_tensor=True).cpu().numpy()


def search_similar_vectors(query, top_k=TOP_K_RESULTS):
    """Search for similar vectors in the FAISS index."""
    query_vector = encode_query(query)
    distances, indices = index.search(query_vector, top_k)
    return distances[0], indices[0]


def main():
    print("Welcome to the FAISS Query System!")
    print(
        f"This system will return the top {TOP_K_RESULTS} most similar results for your query.")
    print("Enter 'quit' to exit the program.")

    while True:
        query = input("\nEnter your query: ")
        # encodeQuery = encode_query(query)
        if query.lower() == 'quit' or query.lower() == 'exit':
            print("Thank you for using the Learner's Mate System. Goodbye!")
            break

        res = index.similarity_search_with_relevance_scores(query, k=5)

        for re in res:
            print(f"Answer:-> {re}")
            print("...............................................")


if __name__ == "__main__":
    main()
