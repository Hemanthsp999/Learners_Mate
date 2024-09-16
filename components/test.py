from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import pipeline
from typing import List
from langchain_core.documents import Document

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

qa_model_name = "distilbert/distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
device = 'cuda'
qa_pipeline = pipeline("question-answering", model=model,
                       tokenizer=tokenizer, device=device)

llm = HuggingFacePipeline(pipeline=qa_pipeline)

template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:"""

prompt = PromptTemplate.from_template(template=template)

retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K_RESULTS})


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = RunnableSequence({
    "context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


def process_llm(answers: str, sources: List[Document]):
    print(f"\nAnswer: {answers}")
    print("\nsources")

    for i, source in enumerate(sources, 1):
        print(f"{i}-{source.page_content[:100]}...")


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
        if query.lower() in ['quit', 'exit']:
            print("Thank you for using the Learner's Mate System. Goodbye!")
            break

        # NOTE :
        # we pass encodeQuery[0] to similarity_search_by_vector().
        # This is because encode_query() returns a 2D array(for batch processing),
        # but we only need the first (and only) vector.

        try:
            answer = rag_chain.invoke(query)
            docs = retriever.get_relevant_documents(query)

            process_llm(answer, docs)

        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
