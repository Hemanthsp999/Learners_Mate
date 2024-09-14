import os
import numpy as np
import faiss
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


model = SentenceTransformer('all-mpnet-base-v2')
model.to(device="cuda")

model_kwargs = {'device': 'cuda'}

encode_kwargs = {'normalize_embeddings': False}


faiss_index_file = "/home/hexa/LearnersMate/faiss_index_file"
sentence_map_file = "sentence_map.pkl"

d = 768

embeddings = HuggingFaceEmbeddings(
    model_name='all-mpnet-base-v2',
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


def Initialize():
    if os.path.exists(faiss_index_file):
        print("Loading file...")
        index = FAISS.load_local(
            faiss_index_file, embeddings, allow_dangerous_deserialization=True)
    else:
        print("File is creating")
        index = faiss.IndexFlatL2(d)
    return index


def embedding_funtion(sentences):
    if isinstance(sentences, str):
        sentences = [sentences]
        '''
    elif isinstance(sentences, dict):
        sentences = [sentences['sentence_chunk']]
        '''
    elif not isinstance(sentences, list):
        raise ValueError(
            f"Expected a list of sentences but got {type(sentences)}")

    embeds = model.encode(sentences, batch_size=32,
                          device='cuda', convert_to_tensor=True)
    embeds = embeds.cpu().numpy()
    # print(f"Embeddings shape: {embeds.shape}")

    if not isinstance(embeds, np.ndarray):
        raise ValueError(
            f"Expected Embeddings to numpy array but got: {type(embeds)}")

    if len(embeds.shape) != 2:
        raise ValueError(
            f"Expected 2D array for embeddings but got : {embeds.shape}")

    return embeds


def Load_Vectors(page: list[dict]) -> FAISS:
    print("Embedding sentences and adding to Faiss Index...")

    docs = [Document(page_content=sentence['sentence_chunks'])
            for sentence in tqdm(page, desc="Processing Sentences")]
    print('Adding to Faiss')
    db = FAISS.from_documents(docs, embeddings)

    return db


def saveIndex(db: FAISS, file_path: str):
    print(f"Saving FAISS index to {file_path}...")
    db.save_local(file_path)


def search_similar_vectors(db: FAISS, query: str, k: int = 5):
    # query_vector = embedding_funtion(query)
    res = db.similarity_search(query, k)
    return res
