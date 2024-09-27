import os
import time
from typing import List
from components import Model
from transformers import pipeline
from werkzeug.utils import secure_filename
from langchain.schema import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from langchain.schema.runnable import RunnablePassthrough
from flask import Flask, render_template, request, jsonify
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)

# NOTE: steps to follow
# 1. PDF Documents.
# 2. Process and make it into sentences.
# 3. Convert Sentences into small chunks of size 10 i.e [sentence1,sentence2...10].
# 4. convert those chunks into embeddings i.e sentence into vectors or tensors.
# 5. Store vectors in FAISS vector database.
# 6. Integrage LLM .
# 7. User Query.

# Configuration
FAISS_INDEX_FILE = "/home/hexa/LearnersMate/faiss_index_file"

# Default PDF
FILE_PATH = "/home/hexa/LearnersMate/PDF_Dataset/MachineLearning.pdf"
MODEL_NAME = 'all-mpnet-base-v2'
TOP_K_RESULTS = 5


# Load the model and FAISS index
print(f"Loading the SentenceTransformer model: {MODEL_NAME}")

print(f"Loading the FAISS index from: {FAISS_INDEX_FILE}...")
embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
if os.path.exists(FAISS_INDEX_FILE):
    print(f"Loading Faiss index file: {FAISS_INDEX_FILE}")
    vector_store = FAISS.load_local(
        FAISS_INDEX_FILE, embeddings=embeddings, allow_dangerous_deserialization=True)
else:
    print(f"Uploaded default file wait to load: {FILE_PATH}")
    start_time = time.time()
    vector_store = Model.upload_file_to_vector(FILE_PATH)
    end_time = time.time()

    print(
        f"Total Time Taken to store vectors: {(end_time - start_time): .2f}s")

# LLM model -> "google/flan-t5-base"
qa_model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(
    qa_model_name, padding=True, truncation=True, max_length=512)
model = AutoModelForSeq2SeqLM.from_pretrained(qa_model_name)

# NOTE: replace to "cpu" if you don't have "cuda" -> "GPU".
device = 'cpu'
qa_pipeline = pipeline("text2text-generation", model=model,
                       tokenizer=tokenizer, device=device)


llm = HuggingFacePipeline(pipeline=qa_pipeline)
template = """
Use the following pieces of context to answer the question at the end. If you don't know the answer or if the question is not related to the given context, please respond with "I'm sorry, but I don't have enough information to answer that question based on the provided context."

Context: {context}

Question: {question}

Answer:
"""
prompt = PromptTemplate.from_template(
    template=template)

# retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K_RESULTS})
retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K_RESULTS})


def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)


rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)


def process_llm(answers: str, sources: List[Document]):
    print(f"\nAnswer: {answers}")
    print("\nsources")

    for i, source in enumerate(sources, 1):
        print(f"{i}-{source.page_content}...")


@app.route("/")
def Home():
    return render_template("navbar.html")


@app.route("/query", methods=["POST"])
def process_query():
    try:
        user_query = request.json.get("query")
        print(f"user query: {user_query}")

        answer = rag_chain.invoke(user_query)
        print(f"answers: {answer}")
        docs = retriever.get_relevant_documents(user_query)

        sources = [doc.page_content for doc in docs]

        if "I don't have enough information to answer this question" in answer:
            relelvance = "Not Relevance"
        else:
            relelvance = "Relevance"

        response = {
            "answer": answer,
            "sources": sources,
            "Relevance": relelvance
        }

        print(f"Response: {response}")

        return jsonify(response)

    except Exception as e:
        print(f"Exception: {str(e)}")
        return jsonify({"error": str(e)}), 500


UPLOAD_FOLDER = '/home/hexa/LearnersMate/TargetFolder/'
# only Pdf and txt format documents are allowed. You can furthur alter this.
Allowed_Extensions = {'pdf', 'txt'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Allowed_Extensions


@app.route("/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file found"}), 400

    file = request.files['file']

    if file.filename == "":
        return jsonify({"error": "No selected files"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        start_time = time.time()
        vector_store = Model.upload_file_to_vector(file_path)
        end_time = time.time()
        print(
            f"Total time taken to upload: {(end_time - start_time): .2f}s")
        vector_store.load_local(
            FAISS_INDEX_FILE, embeddings=embeddings, allow_dangerous_deserialization=True)

        return jsonify({"message": "file uploaded successfully"}), 200


if __name__ == "__main__":
    app.run(debug=True)
