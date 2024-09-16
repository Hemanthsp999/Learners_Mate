from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, DistilBertForQuestionAnswering
from transformers import pipeline

# NOTE: -y: yes & -n: no
# 1. Collect Documents - y
# 2. Clean and Process the Documents into chunks - y
# 3. Convert sentences into Embeddings - y
# 4. Store it in vector DB - y
# 5. then work towards LLM - working on it..

# Configuration
FAISS_INDEX_FILE = "/home/hexa/LearnersMate/faiss_index_file"
MODEL_NAME = 'all-mpnet-base-v2'
TOP_K_RESULTS = 5

# Load the model and FAISS index
print(f"Loading the SentenceTransformer model: {MODEL_NAME}")
st_model = SentenceTransformer(MODEL_NAME)

print(f"Loading the FAISS index from: {FAISS_INDEX_FILE}...")
embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
vector_store = FAISS.load_local(
    FAISS_INDEX_FILE, embeddings=embeddings, allow_dangerous_deserialization=True)

qa_model_name = "distilbert-base-uncased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
model = DistilBertForQuestionAnswering.from_pretrained(qa_model_name)

qa_pipeline = pipeline("question-answering", device="cuda",
                       model=model, tokenizer=tokenizer)

llm = HuggingFacePipeline(pipeline=qa_pipeline)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": TOP_K_RESULTS})
)


def encode_query(query):
    """Encode the query using the SentenceTransformer model."""
    return model.encode([query], convert_to_tensor=True).cpu().numpy()


def main():
    print("Welcome to the FAISS Query System!")
    print(
        f"This system will return the top {TOP_K_RESULTS} most similar results for your query.")
    print("Enter 'quit' to exit the program.")

    while True:
        query = input("\nEnter your query: ")
        # encodeQuery = FaissBase.embeddings.embed_query(query)
        if query.lower() == 'quit' or query.lower() == 'exit':
            print("Thank you for using the Learner's Mate System. Goodbye!")
            break

        # NOTE :
        # we pass encodeQuery[0] to similarity_search_by_vector().
        # This is because encode_query() returns a 2D array(for batch processing),
        # but we only need the first (and only) vector.
        retriver = qa_chain.retriever.get_relevant_documents(query)

        context = "".join([doc.page_content for doc in retriver])
        if not context.strip():
            continue

        result = qa_pipeline({
            "question": query,
            "context": context
        })

        print("\nAnswer:", result['result'])
        print("\nSources")
        for doc in result['source_documents']:
            print(f"- {doc.page_content[:100]}....")


if __name__ == "__main__":
    main()
