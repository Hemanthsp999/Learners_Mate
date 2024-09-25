import os
import shutil
import faiss
from tqdm.auto import tqdm
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


# NOTE: replace to "cpu" if you don't have "cuda" -> "GPU".
model_kwargs = {'device': 'cpu'}

encode_kwargs = {'normalize_embeddings': False}


# Path to store vector database file to conduct RAG operations
faiss_index_file = "/home/hexa/LearnersMate/faiss_index_file"

# dimensions of tensors
d = 768

# embeddings to convert to sentences into some numerical values
embeddings = HuggingFaceEmbeddings(
    model_name='all-mpnet-base-v2',
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


def Initialize_DB():
    if os.path.exists(faiss_index_file):
        print("wait removing file...")
        try:
            print(f"Wait removing {faiss_index_file} directory")
            shutil.rmtree(faiss_index_file)
        except Exception as e:
            print(f"Error occurred while removing the directory: {str(e)}")

    else:
        print("File is creating")
    index = faiss.IndexFlatL2(d)
    return index


# convert sentences to embeddings and stores it in Vector DB
def sentence_to_vectors(page: list[dict]) -> FAISS:
    print("Embedding sentences and adding to Faiss Index...")

    docs = [Document(page_content=sentence['sentence_chunks'])
            for sentence in tqdm(page, desc="Processing Sentences")]
    print('Adding to Faiss...')
    db = FAISS.from_documents(docs, embeddings)

    return db


def saveIndex(db: FAISS, file_path: str):
    print(f"Saving FAISS index to {file_path}...")
    db.save_local(file_path)
    return db


def search_similar_vectors(db: FAISS, query: str, k: int = 5):
    # query_vector = embedding_funtion(query)
    res = db.similarity_search(query, k)
    return res
