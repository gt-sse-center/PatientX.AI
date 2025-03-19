import _pickle
from enum import Enum
from pathlib import Path
import pickle
import sys
from typing import List, Optional, Any

import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from bertopic.vectorizers import ClassTfidfTransformer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.base import ClusterMixin, TransformerMixin
from sklearn.decomposition import PCA
from spacy.lang.en import stop_words
from sentence_transformers import SentenceTransformer
from typing_extensions import Annotated
import typer
from typer_config.decorators import use_yaml_config

from PatientX.models.BERTopicModel import BERTopicModel
from PatientX.utils import read_csv_files_in_directory, get_representation_model
from PatientX.RepresentationModel import RepresentationModel

app = typer.Typer()


class ClusteringModel(str, Enum):
    """
    Enum for clustering algorithm options
    """
    hdbscan = "hdbscan",
    kmeans = "kmeans",
    agglomerative = "agglomerative"


class DimensionalityReduction(str, Enum):
    """
    Enum for dimensionality reduction algorithm options
    """
    umap = "umap",
    pca = "pca"


def get_dimensionality_reduction_model(dim_reduction_model: DimensionalityReduction) -> Optional[TransformerMixin]:
    """
    Get a model instance of the chosen dimensionality reduction algorithm

    :param dim_reduction_model: DimensionalityReductionModel instance - describes the algorithm for dimensionality reduction
    :return: [optional] instance of the chosen dimensionality reduction model or None if it is the default used by bertopic
    """
    match dim_reduction_model:
        case "umap":
            # NOTE: umap returns None since bertopic defaults to a umap model instance
            return None
        case "pca":
            # TODO: update so n_components is not hardcoded
            return PCA(n_components=10)
        case _:
            sys.stdout.write("WARNING: Unknown dimensionality reduction model - defaulting to umap\n")
            return None


def get_clustering_model(clustering_model: ClusteringModel) -> Optional[ClusterMixin]:
    """
    Get a model instance of the chosen clustering algorithm

    :param clustering_model: ClusteringModel instance that describes the clustering algorithm
    :return: [optional] object of the chosen algorithm or None if it is the default used by bertopic
    """
    match clustering_model:
        case "hdbscan":
            # NOTE: hdbscan returns None since bertopic defaults to an hdbscan model instance
            return None
        case "kmeans":
            # TODO: update so n_clusters is not hard coded
            return KMeans(n_clusters=50)
        case "agglomerative":
            # TODO: update so n_clusters is not hard coded
            return AgglomerativeClustering(n_clusters=50)
        case _:
            sys.stdout.write("WARNING: Unknown clustering model - defaulting to hdbscan\n")


def run_bertopic_model(documents: List[str], embeddingspath: Path, result_path: Path,
                       dimensionality_reduction: DimensionalityReduction,
                       clustering_model: ClusteringModel, representationmodel: RepresentationModel, min_topic_size: int,
                       nr_docs: int, document_diversity: float, low_memory: bool,
                       nr_representative_docs: int, prompt: str, api_key: str) -> tuple[

    DataFrame, ndarray | Any, tuple[Any, dict[int, list[tuple[str | list[str], Any] | tuple[str, float]]]]]:
    """
    Run the bertopic model on the given documents with the given model parameters


    :param documents: list of documents to run the bertopic algorithm on
    :param embeddingspath: path to saved embeddings to load
    :param result_path: output path for results
    :param dimensionality_reduction: type of dimensionality reduction algorithm to use
    :param clustering_model: type of clustering algorithm to use
    :param representationmodel: type of representation model to use
    :param min_topic_size: minimum documents in a topic cluster
    :param nr_docs: number of documents to pass into the representation model
    :param document_diversity: document diversity parameter -> float from 0-1
    :param low_memory: low memory flag
    :param nr_representative_docs: Number of representative docs to pass to representation model
    :param prompt: prompt for LLM
    :param api_key: OpenAI API key

    :return: tuple of pd.DataFrame holding results and tensor holding document embeddings

    """
    representation_model = get_representation_model(model_type=representationmodel, nr_docs=nr_docs,
                                                    document_diversity=document_diversity, api_key=api_key,
                                                    prompt=prompt)

    medical_embedding_model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')
    custom_stop_words = list(
        {'with', 'my', 'your', 'she', 'this', 'was', 'her', 'have', 'as', 'he', 'him', 'but', 'not',
         'so', 'are', 'at', 'be', 'has', 'do', 'got', 'how', 'on', 'or', 'would', 'will', 'what',
         'they', 'if', 'or', 'get', 'can', 'we', 'me', 'can', 'has', 'his', 'there', 'them', 'just',
         'am', 'by', 'that', 'from', 'it', 'is', 'in', 'you', 'also', 'very', 'had', 'a', 'an',
         'for'}.union(stop_words.STOP_WORDS))

    vectorizer_model = CountVectorizer(stop_words=custom_stop_words)
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
    dimensionality_reduction_model = get_dimensionality_reduction_model(dimensionality_reduction)
    clustering_model = get_clustering_model(clustering_model)

    bertopic_model = BERTopicModel(ctfidf_model=ctfidf_model, embedding_model=medical_embedding_model, verbose=True,
                                   min_topic_size=min_topic_size, vectorizer_model=vectorizer_model,
                                   representation_model=representation_model, low_memory=low_memory,
                                   hdbscan_model=clustering_model, umap_model=dimensionality_reduction_model,
                                   nr_representative_docs=nr_representative_docs)

    document_embeddings = None
    if embeddingspath.is_file():
        sys.stdout.write("Loading embeddings...\n")
        try:
            document_embeddings = pickle.load(open(embeddingspath, "rb"))
        except _pickle.UnpicklingError:
            sys.stdout.write(
                "WARNING: incorrect embedding file format. Please check README for the proper file format\nGenerating embeddings...\n")
            document_embeddings = medical_embedding_model.encode(documents, show_progress=True)
    else:
        sys.stdout.write("Generating embeddings...\n")
        document_embeddings = medical_embedding_model.encode(documents, show_progress_bar=True)

    sys.stdout.write("Done!\n")

    sys.stdout.write("Fitting Model...\n")
    bertopic_model.fit(documents=documents, embeddings=document_embeddings)

    sys.stdout.write("\nSaving model output...\n")

    # save model output
    # TODO: update loading of bertopic model to work with safetensors approach

    bertopic_model.save(result_path / "bertopic_model_dir", serialization="safetensors", save_ctfidf=True)

    results_df = bertopic_model.get_topic_info()
    rep_docs = results_df['Representative_Docs'].tolist()

    rep_docs_df = pd.DataFrame(rep_docs)
    results_df.drop('Representative_Docs', axis=1, inplace=True)

    results_df = pd.concat([results_df, rep_docs_df], axis=1)

    return results_df, document_embeddings, bertopic_model.get_bertopic_only_results()


def format_bertopic_results(results_df: pd.DataFrame, representative_docs: dict[int, List[str]],
                            bertopic_representative_words: pd.DataFrame) -> pd.DataFrame:
    """
    Take relevant dataframes and return one formatted dataframe holding bertopic intermediate results

    :param results_df: Result of full pipeline
    :param representative_docs: representative docs for each topic
    :param bertopic_representative_words: representative words for each topic
    :return: dataframe containing topics, counts, representative words, representative docs
    """
    get_words = lambda xs: str([x[0] for x in xs])
    bertopic_representative_words = {k: get_words(v) for k, v in bertopic_representative_words.items()}

    counts = results_df['Count']
    topics = results_df['Topic']

    bertopic_results_df = pd.DataFrame.from_dict(representative_docs, orient='index')
    rep_words = pd.DataFrame.from_dict(bertopic_representative_words, orient='index')

    bertopic_final_res = pd.concat(
        [topics.reset_index(), counts.reset_index(), rep_words.reset_index(), bertopic_results_df.reset_index()],
        axis=1)

    return bertopic_final_res


@app.command()
@use_yaml_config()
def main(
        datapath: Annotated[Path, typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=True,
            resolve_path=True,
            help="Path to data - either folder holding CSV files or path to a txt file"
        )] = Path("./data"),
        embeddingspath: Annotated[Path, typer.Option(
            exists=False,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        )] = Path("./output/embeddings.pkl"),
        resultpath: Annotated[Path, typer.Argument(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="Folder to save results in"
        )] = Path("./output"),
        nr_representative_docs: Annotated[int, typer.Option(min=1,
                                                            help="Number of representative docs to save in intermediate Bertopic results")] = 10,
        low_memory: Annotated[bool, typer.Option()] = False,
        nr_docs: Annotated[
            int, typer.Option(min=1, help="Number of representative docs to pass to representation model")] = 10,
        document_diversity: Annotated[float, typer.Option()] = 0.1,
        representationmodel: Annotated[
            RepresentationModel, typer.Option(case_sensitive=False)] = RepresentationModel.mistral_small,
        prompt: Annotated[str, typer.Option()] = None,
        min_topic_size: Annotated[int, typer.Option()] = 100,
        clustering_model: Annotated[ClusteringModel, typer.Option(case_sensitive=False)] = ClusteringModel.hdbscan,
        dimensionality_reduction: Annotated[
            DimensionalityReduction, typer.Option(case_sensitive=False)] = DimensionalityReduction.umap,
        save_embeddings: Annotated[bool, typer.Option()] = False,
        api_key: Annotated[str, typer.Option(help="OpenAI API key")] = None
):
    datapath = Path(datapath)
    resultpath = Path(resultpath)
    embeddingspath = Path(embeddingspath)

    sys.stdout.write("Reading data...\n")
    documents = read_csv_files_in_directory(datapath)

    if len(documents) == 0:
        sys.stdout.write("No data found\n")
        return

    sys.stdout.write("Done!\n")

    results_df, document_embeddings, bertopic_only_results = run_bertopic_model(documents=documents,
                                                                                embeddingspath=embeddingspath,
                                                                                result_path=resultpath,
                                                                                dimensionality_reduction=dimensionality_reduction,
                                                                                clustering_model=clustering_model,
                                                                                representationmodel=representationmodel,
                                                                                min_topic_size=min_topic_size,
                                                                                nr_docs=nr_docs,
                                                                                document_diversity=document_diversity,
                                                                                low_memory=low_memory,
                                                                                nr_representative_docs=nr_representative_docs,
                                                                                prompt=prompt, api_key=api_key)
    results_df.to_csv(resultpath / "output.csv", index=False)

    representative_docs, bertopic_representative_words = bertopic_only_results

    bertopic_final_res = format_bertopic_results(results_df, representative_docs, bertopic_representative_words)
    bertopic_final_res.to_csv(resultpath / "bertopic_final_results.csv", index=False)

    if save_embeddings:
        pickle.dump(document_embeddings, open(resultpath / "embeddings.pkl", "wb"))


if __name__ == '__main__':
    app()
