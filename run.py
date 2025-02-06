import csv
from enum import Enum
from pathlib import Path
import pickle
from typing import List, Optional

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.base import ClusterMixin, TransformerMixin
from sklearn.decomposition import PCA
from spacy.lang.en import stop_words
from sentence_transformers import SentenceTransformer
from typing_extensions import Annotated
import typer
from typer_config.decorators import use_yaml_config

from PatientX.models.BERTopicModel import BERTopicModel
from PatientX.MistralRepresentation import MistralRepresentation


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


def read_csv_files_in_directory(datafolder: Path) -> Optional[List[str]]:
    """
    Read in data from all CSV files in directory
    Expected data format of CSV files in README


    :param datafolder: Path
    :raises NotADirectoryError: If datafolder is not a directory
    :raises KeyError: If data does not match structure found in README
    :return: optional List<str> containing documents, None if no valid csv files in directory
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
            except csv.Error as e:
                print(f"{filename}: is not a CSV file. File ignored")
                pass

    if len(dfs) == 0:
        return None

    full_dataset = pd.concat(dfs, ignore_index=True)
    cleaned_text = pd.DataFrame()
    try:
        grouped_dataset = full_dataset.groupby(['forum', 'thread_title', 'message_nr'], as_index=False).agg(
            {'post_message': ''.join})

        grouped_dataset['post_message'] = grouped_dataset['post_message'].str.strip().replace(r'\n', ' ', regex=True)
        cleaned_text = grouped_dataset['post_message'].replace({r'\s+$': '', r'^\s+': ''}, regex=True).replace(r'\n',
                                                                                                               ' ',
                                                                                                               regex=True)
        cleaned_text = cleaned_text.replace(r'\t', ' ', regex=True)
        cleaned_text = cleaned_text.replace(r'\r', ' ', regex=True)
    except KeyError as e:
        raise KeyError("Check README file for proper data format")

    return cleaned_text.tolist()


@app.command()
@use_yaml_config()
def main(
        datapath: Annotated[Path, typer.Option()] = "./data",
        embeddingspath: Annotated[Path, typer.Option()] = "./embeddings",
        resultpath: Annotated[Path, typer.Option()] = "./output",
        low_memory: Annotated[bool, typer.Option()] = False,
        nr_docs: Annotated[int, typer.Option()] = 10,
        document_diversity: Annotated[float, typer.Option()] = 0.1,
        prompt: Annotated[str, typer.Option()] = None,
        min_topic_size: Annotated[int, typer.Option()] = 100,
        clustering_model: Annotated[ClusteringModel, typer.Option(case_sensitive=False)] = ClusteringModel.hdbscan,
        dimensionality_reduction: Annotated[
            DimensionalityReduction, typer.Option(case_sensitive=False)] = DimensionalityReduction.umap,
        save_embeddings: Annotated[bool, typer.Option()] = False,
):
    print("Reading data...")
    documents = read_csv_files_in_directory(datapath)

    if not documents:
        print("No data found")
        return

    print("Done!")

    representation_model = MistralRepresentation(nr_docs=nr_docs, diversity=document_diversity, api="generate")
    medical_embedding_model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')
    custom_stop_words = {'with', 'my', 'your', 'she', 'this', 'was', 'her', 'have', 'as', 'he', 'him', 'but', 'not',
                         'so', 'are', 'at', 'be', 'has', 'do', 'got', 'how', 'on', 'or', 'would', 'will', 'what',
                         'they', 'if', 'or', 'get', 'can', 'we', 'me', 'can', 'has', 'his', 'there', 'them', 'just',
                         'am', 'by', 'that', 'from', 'it', 'is', 'in', 'you', 'also', 'very', 'had', 'a', 'an',
                         'for'}.union(stop_words.STOP_WORDS)

    vectorizer_model = CountVectorizer(stop_words=custom_stop_words)
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
    dimensionality_reduction_model = get_dimensionality_reduction_model(dimensionality_reduction)
    clustering_model = get_clustering_model(clustering_model)

    bertopic_model = BERTopicModel(ctfidf_model=ctfidf_model, embedding_model=medical_embedding_model, verbose=True,
                                   min_topic_size=min_topic_size, vectorizer_model=vectorizer_model,
                                   representation_model=representation_model, low_memory=low_memory,
                                   hdbscan_model=clustering_model, umap_model=dimensionality_reduction_model)

    document_embeddings = None
    if embeddingspath.is_file():
        print("Loading embeddings...")
        document_embeddings = pickle.load(open(embeddingspath, "rb"))
    else:
        document_embeddings = medical_embedding_model.encode(documents, show_progress_bar=True)
        print("Generating embeddings...")

    print("Done!")

    if save_embeddings:
        pickle.dump(document_embeddings, open(resultpath / "embeddings.pkl", "wb"))

    print("Fitting model...")
    bertopic_model = bertopic_model.fit(documents=documents, embeddings=document_embeddings)
    bertopic_model.transform(documents, embeddings=document_embeddings)

    print("Saving model output...")

    # save model output
    results_df = bertopic_model.get_topic_info()
    rep_docs = results_df['Representative_Docs'].tolist()

    rep_docs_df = pd.DataFrame(rep_docs)
    results_df.drop('Representative_Docs', axis=1, inplace=True)

    results_df = pd.concat([results_df, rep_docs_df], axis=1)
    results_df.to_csv(resultpath / "output.csv", index=False)


if __name__ == '__main__':
    app()
