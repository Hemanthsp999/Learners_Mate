import os
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
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

app = Flask(__name__)

# NOTE: -y: yes & -n: no
# 1. Collect Documents - y
# 2. Clean and Process the Documents into chunks - y
# 3. Convert sentences into Embeddings - y
# 4. Store it in vector DB - y
# 5. LLM completed - y

# Configuration
FAISS_INDEX_FILE = "/home/hexa/LearnersMate/faiss_index_file"
MODEL_NAME = 'all-mpnet-base-v2'
TOP_K_RESULTS = 5

# Load the model and FAISS index
print(f"Loading the SentenceTransformer model: {MODEL_NAME}")

print(f"Loading the FAISS index from: {FAISS_INDEX_FILE}...")
embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
vector_store = FAISS.load_local(
    FAISS_INDEX_FILE, embeddings=embeddings, allow_dangerous_deserialization=True)

# Here i'm using distilbert/distilbert-base-uncased model
qa_model_name = "distilbert/distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
device = 'cuda'
qa_pipeline = pipeline("question-answering", model=model,
                       tokenizer=tokenizer, device=device)


llm = HuggingFacePipeline(pipeline=qa_pipeline, model_kwargs={
                          "max_length": 512, "truncation": True})

template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Key Points: {context}

Answer: """

prompt = PromptTemplate.from_template(template)

retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K_RESULTS})


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


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

        answer = rag_chain.invoke(user_query)
        docs = retriever.get_relevant_documents(user_query)
        sources = [doc.page_content for doc in docs]
        response = {
            "answer": answer,
            "sources": sources
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


UPLOAD_FOLDER = '/home/hexa/LearnersMate/Dataset'
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

        Model.upload_file_to_vector(file_path)

        return jsonify({"message": "file uploaded successfully"}), 200


if __name__ == "__main__":
    app.run(debug=True)
