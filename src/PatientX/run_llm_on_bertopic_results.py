from typing import Mapping, List, Tuple

import pandas as pd
from pathlib import Path
import typer
from typing_extensions import Annotated

from PatientX.models.MistralRepresentation import MistralRepresentation
from PatientX.RepresentationModel import RepresentationModel
from llm_utils import load_bertopic_model_from_pkl

DEFAULT_PROMPT = """
I have topic that contains the following documents: \n[DOCUMENTS]
The topic is described by the following keywords: [KEYWORDS]

Based on the above information, can you give a short label of the topic?
"""

app = typer.Typer()


@app.command()
def run_llm_on_bertopic_results(documents: pd.DataFrame, bertopic_results_filepath: Annotated[Path, typer.Option(
    exists=True,
    file_okay=True,
    dir_okay=False,
    resolve_path=True,
)], llm_option: Annotated[RepresentationModel, typer.Option(case_sensitive=False)], prompt: str = DEFAULT_PROMPT) -> \
Mapping[
    str, List[Tuple[str, float]]]:
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


if __name__ == '__main__':
    app()
