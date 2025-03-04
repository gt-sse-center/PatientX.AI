import csv
import sys
from pathlib import Path
from typing import List
import pandas as pd

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