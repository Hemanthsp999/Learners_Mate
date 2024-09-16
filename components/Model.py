import fitz
import pandas as pd
import os
import TextProcess
import FaissBase
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


def main():

    dirPath = "/home/hexa/LearnersMate/PDF_Dataset/"
    # save chunks to csv file to track the sentences
    csvPath = "sentence.csv"

    if not os.path.exists(dirPath):
        raise ValueError("Dir Path not exists. Check once")

    index = FaissBase.Initialize_DB()
    print(f"Index initialized successfully: {index}")
    for file_path in os.listdir(dirPath):
        file = os.path.join(dirPath, file_path)
        if not os.path.isfile(file):
            raise ValueError("Something Error, this is not a file")

        docs_to_dict = open_and_process(file)

        getSentences = TextProcess.list_to_sentence(
            dict_to_sentence=docs_to_dict)

        splitSentence = TextProcess.split_sentence(sentence=getSentences)

        sentence_chunks = TextProcess.Refactor_sentence(
            sentences=splitSentence)

        df = pd.DataFrame(sentence_chunks)

        TextProcess.get_token_length(dataframe=df)

        pages_and_chunks = df[df['chunk_token_count']
                              > 30].to_dict(orient="records")
        db = FaissBase.Load_Vectors(pages_and_chunks)

        df = pd.DataFrame(pages_and_chunks)

        df.to_csv(csvPath, mode='a', index=False)
    print(df.describe().round(2))
    FaissBase.saveIndex(db=db, file_path=FaissBase.faiss_index_file)

    query = "what is Process Model?"
    distance = FaissBase.search_similar_vectors(db, query)
    print(f"Answer: {distance[0].page_content}")


if __name__ == "__main__":
    main()
