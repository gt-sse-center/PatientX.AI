import pandas as pd
from pathlib import Path
import typer
from typing_extensions import Annotated

from PatientX.RepresentationModel import RepresentationModel
from PatientX.utils import load_bertopic_model_from_pkl, get_representation_model

DEFAULT_PROMPT = """
I have topic that contains the following documents: \n[DOCUMENTS]
The topic is described by the following keywords: [KEYWORDS]

Based on the above information, can you give a short label of the topic?
"""

app = typer.Typer()

@app.command()
def run_llm_on_bertopic_results_csv(bertopic_results_filepath: Annotated[Path, typer.Option(
    exists=True,
    file_okay=True,
    dir_okay=False,
    resolve_path=True,
)], output_dir: Annotated[Path, typer.Option(
    exists=True,
    file_okay=False,
    dir_okay=True,
    resolve_path=True,
)], llm_option: Annotated[RepresentationModel, typer.Option(case_sensitive=False)], prompt: str = DEFAULT_PROMPT):
    """
        Run the selected llm on top of bertopic results. Requires a saved bertopic model in pkl format

        :param output_dir: folder to store results
        :param bertopic_results_filepath: filepath to pkl file holding bertopic model that was trained on passed in documents
        :param llm_option: mistral-small or gpt4o
        :param prompt: prompt to pass into LLM. Use [DOCUMENTS] and [KEYWORDS] as placeholders for documents and keywords
        """
    bertopic_model = load_bertopic_model_from_pkl(bertopic_results_filepath)
    representation_model = get_representation_model(llm_option, prompt=prompt)

    llm_results = representation_model.extract_topics(bertopic_model, documents=bertopic_model.fit_documents,
                                               c_tf_idf=bertopic_model.c_tf_idf_,
                                               topics=bertopic_model.bertopic_representative_words)

    llm_results_df = pd.DataFrame.from_dict(llm_results, orient='index')
    llm_results_df.to_csv(output_dir / 'llm_results.csv')


if __name__ == '__main__':
    app()
