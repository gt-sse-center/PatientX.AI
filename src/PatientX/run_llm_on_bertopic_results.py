from typing import Mapping, List, Tuple

import pandas as pd
from pathlib import Path
import typer
from typing_extensions import Annotated

from PatientX.models.MistralRepresentation import MistralRepresentation
from PatientX.RepresentationModel import RepresentationModel
from llm_utils import load_bertopic_model_from_pkl
from utils import read_csv_files_in_directory

DEFAULT_PROMPT = """
I have topic that contains the following documents: \n[DOCUMENTS]
The topic is described by the following keywords: [KEYWORDS]

Based on the above information, can you give a short label of the topic?
"""

app = typer.Typer()

def run_llm_on_bertopic_results_dataframe(documents: pd.DataFrame, bertopic_results_filepath: Annotated[Path, typer.Option(
    exists=True,
    file_okay=True,
    dir_okay=False,
    resolve_path=True,
)], llm_option: Annotated[RepresentationModel, typer.Option(case_sensitive=False)], prompt: str = DEFAULT_PROMPT) -> \
Mapping[
    str, List[Tuple[str, float]]]:
    """
    Run the selected llm on top of bertopic results. Requires a saved bertopic model in pkl format

    :param documents: dataframe holding documents
    :param bertopic_results_filepath: filepath to pkl file holding bertopic model that was trained on passed in documents
    :param llm_option: mistral-small or gpt4o
    :param prompt: prompt to pass into LLM. Use [DOCUMENTS] and [KEYWORDS] as placeholders for documents and keywords
    """
    bertopic_model = load_bertopic_model_from_pkl(bertopic_results_filepath)
    representation_model = None

    match llm_option:
        case 'mistral':
            representation_model = MistralRepresentation(prompt=prompt)
        case 'gpt-4o':
            # TODO: update to use gpt
            representation_model = MistralRepresentation(prompt=prompt)
            pass

    return representation_model.extract_topics(bertopic_model, documents=documents, c_tf_idf=bertopic_model.c_tf_idf,
                                               topics=bertopic_model.topics_)


@app.command()
def run_llm_on_bertopic_results_csv(datapath: Annotated[Path, typer.Option(
    exists=True,
    file_okay=False,
    dir_okay=True,
    resolve_path=True,
)], bertopic_results_filepath: Annotated[Path, typer.Option(
    exists=True,
    file_okay=True,
    dir_okay=False,
    resolve_path=True,
)], llm_option: Annotated[RepresentationModel, typer.Option(case_sensitive=False)], prompt: str = DEFAULT_PROMPT):
    documents = read_csv_files_in_directory(datapath)
    documents = pd.DataFrame(documents)
    run_llm_on_bertopic_results_dataframe(documents=documents, bertopic_results_filepath=bertopic_results_filepath, llm_option=llm_option, prompt=prompt)


if __name__ == '__main__':
    app()
