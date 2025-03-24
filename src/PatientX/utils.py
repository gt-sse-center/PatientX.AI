import csv
import sys
from pathlib import Path
from typing import List

from PatientX.models.MistralRepresentation import MistralRepresentation
from PatientX.RepresentationModelEnum import RepresentationModel

from bertopic import BERTopic
from bertopic.representation import OpenAI
import numpy as np
import openai
import pandas as pd


def read_data_in_txt_file(filepath: Path) -> List[str]:
    """
    Read data in from txt file

    :param filepath: path to txt file
    :return: list of documents (each new line is treated as a document)
    """
    if filepath.is_file():
        with filepath.open() as f:
            return f.readlines()

    return []


def read_csv_files_in_directory(datafolder: Path) -> List[str]:
    """
    Read in data from all CSV files in directory
    Expected data format of CSV files in README


    :param datafolder: Path
    :raises NotADirectoryError: If datafolder is not a directory
    :raises KeyError: If data does not match structure found in README
    :return: List<str> containing documents, [] if no valid csv files in directory
    """
    dfs = []

    if not datafolder.is_dir():
        raise NotADirectoryError("Data folder doesn't exist. Please check the filepath")

    for filename in datafolder.iterdir():
        filepath = datafolder / filename
        if filepath.is_file():
            try:
                df = pd.read_csv(filepath)
                dfs.append(df.copy(deep=True))
            except csv.Error:
                sys.stdout.write(f"WARNING: {filename} is not a CSV file. File ignored\n")

    if len(dfs) == 0:
        return []

    full_dataset = pd.concat(dfs, ignore_index=True)
    try:
        grouped_dataset = full_dataset.groupby(['forum', 'thread_title', 'message_nr'], as_index=False).agg(
            {'post_message': ''.join})

        grouped_dataset['post_message'] = grouped_dataset['post_message'].str.strip().replace(r'\n', ' ', regex=True)
        cleaned_text = grouped_dataset['post_message'].replace({r'\s+$': '', r'^\s+': ''}, regex=True).replace(r'\n',
                                                                                                               ' ',
                                                                                                               regex=True)
        cleaned_text = cleaned_text.replace(r'\t', ' ', regex=True)
        cleaned_text = cleaned_text.replace(r'\r', ' ', regex=True)
    except KeyError:
        raise KeyError("Check README file for proper data format")

    return cleaned_text.tolist()


def read_data(filepath: Path) -> List[str]:
    """
    Read in data from directory/txt file. If a directory is given, data will be parsed from all csv files in the directory. If a txt file is given, each new line will be treated as it's own document
    :param filepath: path to directory or file to read data in from
    :return: list of documents
    """
    if filepath.is_dir():
        sys.stdout.write("Reading data from directory...\n")
        return read_csv_files_in_directory(filepath)
    elif filepath.is_file() and filepath.suffix == '.txt':
        sys.stdout.write("Reading data from txt file...\n")
        return read_data_in_txt_file(filepath)

    sys.stdout.write("ERROR: Incorrect data format - please check README for proper data format\n")
    return []


def load_bertopic_model_from_pkl(filepath: Path):
    """

    :param filepath: path to pkl file
    :return: loaded model or None if filepath does not exist
    """
    if filepath.exists():
        return BERTopic.load(str(filepath))

    return None


def get_representation_model(model_type: RepresentationModel, nr_docs: int = 10, document_diversity: float = 0.1,
                             api_key: str = None, openai_model_name="gpt-4o", prompt=None, chat_flag=False):
    """
    Get an instance of the chosen representation model

    :param api_key: Open AI API key
    :param model_type: Representation model enum
    :param nr_docs: number of docs to pass into the representation model
    :param document_diversity: document diversity parameter for choosing docs to passing to the representation model
    :param chat_flag: True if chat, False if generate
    :return: instance of the chosen representation model
    """

    chat_or_generate = "chat" if chat_flag else "generate"
    match model_type:
        case "mistral-small":
            return MistralRepresentation(nr_docs=nr_docs, diversity=np.clip(document_diversity, 0, 1),
                                         api=chat_or_generate, prompt=prompt)
        case "gpt4o":
            client = openai.OpenAI(api_key=api_key)

            return OpenAI(client=client, model=openai_model_name, prompt=prompt, chat=True)
        case _:
            return None
