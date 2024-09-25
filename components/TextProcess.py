import re
import pandas as pd
from tqdm.auto import tqdm
from spacy.lang.en import English

nlp = English()
nlp.add_pipe("sentencizer")


def list_to_sentence(dict_to_sentence: list[dict]) -> list[dict]:

    for page in tqdm(dict_to_sentence, desc="Processing sentences..."):

        page['sentence'] = list(nlp(page["Text"]).sents)

        page['sentence'] = [str(sentence) for sentence in page['sentence']]

        page['str_sentence_count'] = len(page['sentence'])

    return dict_to_sentence


# Adjust as you need
chunk_size = 5


def split_list(inputList: list[str], split_size: int) -> list[list[str]]:
    return [inputList[i: i+split_size] for i in range(0, len(inputList),
                                                      split_size)]


def split_sentence(sentence: list[str]) -> list[dict]:

    for page in tqdm(sentence):
        page['sentence_chunks'] = split_list(
            page['sentence'], split_size=chunk_size)

        page['number_of_chunks'] = len(page['sentence_chunks'])

    return sentence


def sentence_to_chunks(sentences: list[dict]) -> list[dict]:
    sentence_and_chunks = []

    for page in tqdm(sentences):
        for sentence_chunk in tqdm(page['sentence_chunks'], desc="Processing sentence_chunks"):
            chunk_list = {}
            chunk_list["page_no"] = page.get('page_no')
            joined_chunks = "".join(sentence_chunk).replace(" ", " ").strip()

            joined_chunks = re.sub(r'\.([A-Z|0-9])', r'. \1', joined_chunks)
            chunk_list['sentence_chunks'] = joined_chunks

            chunk_list['chunk_char_count'] = len(joined_chunks)
            chunk_list['chunk_word_count'] = len(
                [word for word in joined_chunks.split(" ")])

            chunk_list['chunk_token_count'] = len(joined_chunks)/4
            sentence_and_chunks.append(chunk_list)

    return sentence_and_chunks


min_token_length = 30


def get_token_length(dataframe: pd.DataFrame):

    for row in dataframe[dataframe['chunk_token_count'] <= min_token_length].sample(5, replace=True).iterrows():
        print(
            f"chunk_token count: {row[1]['chunk_token_count']} | Text: {row[1]['sentence_chunks']}")
