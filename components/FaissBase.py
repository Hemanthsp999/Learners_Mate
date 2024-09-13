from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

model_path = "sentence-transformers/all-MiniLM-l6-v2"

model_kwargs = {'device': 'cuda'}

encode_kwargs = {'normalize_embeddings': False}


embeddings = HuggingFaceEmbeddings(
    model_name=model_path,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


def loadVectors(sentences: list[dict]):
    pages = []

    pages = [Document(page_content=page['sentence_chunks'])
             for page in sentences]

    db = FAISS.from_documents(pages, embeddings)
    print(f"This is vector {db}")

    res = db.similarity_search("what is Machine Learning", k=4)
    for re in res:
        print(f"* {re.page_content}")
