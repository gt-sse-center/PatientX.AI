import _pickle
from enum import Enum
from pathlib import Path
import pickle
import sys
from typing import Optional

import pandas as pd
import numpy as np
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
from PatientX.models.MistralRepresentation import MistralRepresentation
from PatientX.utils import read_csv_files_in_directory


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

def get_representation_model(type, nr_docs, document_diversity):
    match type:
        case "mistral-small":
            return MistralRepresentation(nr_docs=nr_docs, diversity=np.clip(document_diversity, 0, 1),
                                                     api="generate")
        case _:
            return None

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

def run_bertopic_model(documents, embeddingspath, dimensionality_reduction, clustering_model, representationmodel, min_topic_size, nr_docs, document_diversity, low_memory):
    representation_model = get_representation_model(type=representationmodel, nr_docs=nr_docs,
                                                    document_diversity=document_diversity)

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
                                   hdbscan_model=clustering_model, umap_model=dimensionality_reduction_model)

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
    bertopic_model = bertopic_model.fit(documents=documents, embeddings=document_embeddings)
    bertopic_model.transform(documents, embeddings=document_embeddings)

    sys.stdout.write("Saving model output...\n")

    # save model output
    results_df = bertopic_model.get_topic_info()
    rep_docs = results_df['Representative_Docs'].tolist()

    rep_docs_df = pd.DataFrame(rep_docs)
    results_df.drop('Representative_Docs', axis=1, inplace=True)

    results_df = pd.concat([results_df, rep_docs_df], axis=1)

    return results_df, document_embeddings,


@app.command()
@use_yaml_config()
def main(
        datapath: Annotated[Path, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        )] = Path("./data"),
        embeddingspath: Annotated[Path, typer.Option(
            exists=False,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        )] = Path("./output/embeddings.pkl"),
        resultpath: Annotated[Path, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        )] = Path("./output"),
        low_memory: Annotated[bool, typer.Option()] = False,
        nr_docs: Annotated[int, typer.Option()] = 10,
        document_diversity: Annotated[float, typer.Option()] = 0.1,
        representationmodel: Annotated[str, typer.Option()] = "None",
        prompt: Annotated[str, typer.Option()] = None,
        min_topic_size: Annotated[int, typer.Option()] = 100,
        clustering_model: Annotated[ClusteringModel, typer.Option(case_sensitive=False)] = ClusteringModel.hdbscan,
        dimensionality_reduction: Annotated[
            DimensionalityReduction, typer.Option(case_sensitive=False)] = DimensionalityReduction.umap,
        save_embeddings: Annotated[bool, typer.Option()] = False,
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

    results_df, document_embeddings = run_bertopic_model(documents=documents, embeddingspath=embeddingspath, dimensionality_reduction=dimensionality_reduction, clustering_model=clustering_model,representationmodel=representationmodel, min_topic_size=min_topic_size, low_memory=low_memory,nr_docs=nr_docs, document_diversity=document_diversity)
    results_df.to_csv(resultpath / "output.csv", index=False)

    if save_embeddings:
        pickle.dump(document_embeddings, open(resultpath / "embeddings.pkl", "wb"))


if __name__ == '__main__':
    app()
