import argparse
from typing import List, Union
from pathlib import Path
import os
import pickle
import yaml

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from spacy.lang.en import stop_words
from sentence_transformers import SentenceTransformer

from PatientX.models.BERTopicModel import BERTopicModel
from PatientX.MistralRepresentation import MistralRepresentation


def read_data(datafolder: Path):
    """
    Read in data

    datafolder: Path
    Returns: List containing documents
    """
    dfs = []

    # TODO: assert safety

    for filename in os.listdir(datafolder):
        filepath = os.path.join(datafolder, filename)
        if os.path.isfile(filepath):
            df = pd.read_csv(filepath)
            dfs.append(df.copy(deep=True))

    full_dataset = pd.concat(dfs, ignore_index=True)
    grouped_dataset = full_dataset.groupby(['forum', 'thread_title', 'message_nr'], as_index=False).agg(
        {'post_message': ''.join})

    grouped_dataset['post_message'] = grouped_dataset['post_message'].str.strip().replace(r'\n', ' ', regex=True)
    cleaned_text = grouped_dataset['post_message'].replace({r'\s+$': '', r'^\s+': ''}, regex=True).replace(r'\n', ' ',
                                                                                                           regex=True)
    cleaned_text = cleaned_text.replace(r'\t', ' ', regex=True)
    cleaned_text = cleaned_text.replace(r'\r', ' ', regex=True)

    return cleaned_text.tolist()


def get_count_vectorizer():
    custom_stop_words = list(stop_words.STOP_WORDS)
    custom_stop_words += ['with', 'my', 'your', 'she', 'this', 'was', 'her', 'have', 'as', 'he', 'him', 'but', 'not',
                          'so', 'are', 'at', 'be', 'has', 'do', 'got', 'how', 'on', 'or', 'would', 'will', 'what',
                          'they', 'if', 'or', 'get', 'can', 'we', 'me', 'can', 'has', 'his', 'there', 'them', 'just',
                          'am', 'by', 'that', 'from', 'it', 'is', 'in', 'you', 'also', 'very', 'had', 'a', 'an', 'for']

    vectorizer_model = CountVectorizer(stop_words=custom_stop_words)

    return vectorizer_model


def get_ctidf_model():
    return ClassTfidfTransformer(reduce_frequent_words=True)


def get_medical_embedding_model():
    return SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')


def get_representation_model(nr_docs=10, document_diversity=0.1):
    representation_model = MistralRepresentation(nr_docs=nr_docs, diversity=document_diversity, api="chat")

    return representation_model


def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    parser = argparse.ArgumentParser(description="A script with command-line arguments.")
    parser.add_argument("-d", "--datapath", type=str, default=config.get("datapath", "./data/"), help="Path to data")
    parser.add_argument("-e", "--embeddings", type=str, default=config.get("embeddingspath", "./data/embeddings/"), help="Path to data embeddings")
    parser.add_argument("-r", "--result_path", type=str, default=config.get("resultspath", "./output/"), help="Path to save results")
    # parser.add_argument("-im", "--images", action="store_true", help="Save visualizations")
    parser.add_argument("-lm", "--low_memory", action="store_true", help="Low memory flag")
    parser.add_argument("-n", "--nr_docs", type=int, default=config.get("nr_docs", 10),
                        help="Number of docs per topic to pass to representation model")
    parser.add_argument("-p", "--prompt", type=str, default=config.get("prompt", ""), help="Prompt to pass to LLM")
    parser.add_argument("-s", "--min_topic_size", type=int, default=config.get("min_topic_size", 100), help="Minimum topic size")
    #
    # # bertopic options
    #
    parser.add_argument("-cl", "--clustering", type=str, choices=['hbdscan', 'kmeans', 'agglomerative'],
                        default=config.get("clustering_model", "hdbscan"), help="Clustering algorithm")
    # parser.add_argument("-llm", "--llm", type=str, choices=['gpt4o', 'mistral-small'], help="Low memory flag")
    parser.add_argument("-dim", "--dim_reduction", type=str, choices=['pca', 'umap'], default=config.get("dimensionality_reduction", "umap"),
                        help="Dimensionality reduction algorithm")

    args = parser.parse_args()

    documents = read_data(args.datapath)
    representation_model = get_representation_model(nr_docs=args.nr_docs)
    medical_embedding_model = get_medical_embedding_model()
    vectorizer_model = get_count_vectorizer()
    ctfidf_model = get_ctidf_model()

    bertopic_model = BERTopicModel(ctfidf_model=ctfidf_model, embedding_model=medical_embedding_model, verbose=True,
                                   min_topic_size=args.min_topic_size, vectorizer_model=vectorizer_model,
                                   representation_model=representation_model)

    print("embeddings being generated")
    document_embeddings = medical_embedding_model.encode(documents, show_progress_bar=True)
    # if args.embeddings is None:
    #     document_embeddings = medical_embedding_model.encode(dataset)
    #
    #     # TODO: save embeddings
    #     pickle.dump(document_embeddings, open(args.result_path, "wb"))


    print("model fit and transform")
    # bertopic_model = bertopic_model.fit(documents=documents, embeddings=document_embeddings)
    # bertopic_model.transform(documents, embeddings=document_embeddings)


    print("saving model output")
    # save model output
    # results_df = bertopic_model.get_topic_info()
    #
    # rep_docs = results_df['Representative_Docs'].tolist()
    #
    # rep_docs_df = pd.DataFrame(rep_docs)
    # results_df.drop('Representative_Docs', axis=1, inplace=True)
    #
    # results_df = pd.concat([results_df, rep_docs_df], axis=1)
    # results_df.to_csv(args.result_path, index=False)


if __name__ == '__main__':
    main()









