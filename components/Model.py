import fitz
import pandas as pd
import os
# import TextProcess
# import FaissBase
from components import TextProcess
from components import FaissBase
from tqdm.auto import tqdm


def text_formater(text: str) -> str:
    full = "".join(text)
    clean_text = full.replace("\n", " ").strip()
    return clean_text


def open_and_process(pdf: str) -> list[dict]:
    pdf = fitz.open(pdf)
    docs_to_dict = []

    for page_no, page in tqdm(enumerate(pdf), total=len(pdf)):
        text = page.get_text()
        text = text_formater(text)
        docs_to_dict.append({
            "page_no": page_no,
            "char_count": len(text),
            "word_count": len(text.split(" ")),
            "sentence_count": len(text.split(". ")),
            "Token_size": len(text)/4,
            "Text": text
        })

    return docs_to_dict


# Default load this pdf or you can choose your pdf path
file_path = "/home/hexa/LearnersMate/PDF_Dataset/MachineLearning.pdf"


def upload_file_to_vector(file_path):

    # store sentences in csv for debugging purpose
    csvPath = "sentence.csv"

    # check if file path exists or not
    if not os.path.exists(file_path):
        raise ValueError("File path does not exists. Check once")

    # Initialize Vector Database
    index = FaissBase.Initialize_DB()
    print(f"Index initialized successfully: {index}")
    if not os.path.isfile(file_path):
        raise ValueError("Something Error, this is not a file")

    docs_to_dict = open_and_process(file_path)

    getSentences = TextProcess.list_to_sentence(
        dict_to_sentence=docs_to_dict)

    splitSentence = TextProcess.split_sentence(sentence=getSentences)

    sentence_chunks = TextProcess.sentence_to_chunks(
        sentences=splitSentence)

    df = pd.DataFrame(sentence_chunks)

    TextProcess.get_token_length(dataframe=df)

    pages_and_chunks = df[df['chunk_token_count']
                          > 30].to_dict(orient="records")
    db = FaissBase.sentence_to_vectors(pages_and_chunks)

    df = pd.DataFrame(pages_and_chunks)

    df.to_csv(csvPath, mode='a', index=False)
    print(df.describe().round(2))
    vector_store = FaissBase.saveIndex(
        db=db, file_path=FaissBase.faiss_index_file)

    # You can use this for wehter the DB is working as expected or not.
    query = "what is Process Model?"
    distance = FaissBase.search_similar_vectors(db, query)
    print(f"Answer: {distance[0].page_content}")

    return vector_store


# NOTE:
''' uncomment the below comment before running the project and run once at the starting '''
''' And run the Model.py to create faiss_index_file directory '''
# upload_file_to_vector(file_path)
