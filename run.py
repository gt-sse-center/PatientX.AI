import argparse
import csv
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


def read_csv_files_in_directory(datafolder: Path):
    """
    Read in data from all CSV files in directory
    Expected data format of CSV files in README

    :param datafolder: Path
    :return: List<str> containing documents
    """
    dfs = []

    assert (os.path.isdir(datafolder))

    for filename in datafolder.iterdir():
        filepath = datafolder / filename
        if filepath.is_file():
            try:
                df = pd.read_csv(filepath)
                dfs.append(df.copy(deep=True))
            except csv.Error as e:
                print(f"{filename}: is not a CSV file. File ignored")
                pass

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


def main():
    DOCUMENT_DIVERSITY = 0.1
    if not os.path.isfile("config.yaml"):
        raise Exception("Missing config yaml file")

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    parser = argparse.ArgumentParser(description="A script with command-line arguments.")
    parser.add_argument("-d", "--datapath", type=str, default=config.get("datapath", "./data/"), help="Path to data")
    parser.add_argument("-e", "--embeddings", type=str, default=config.get("embeddingspath", "./data/embeddings/"),
                        help="Path to data embeddings")
    parser.add_argument("-r", "--result_path", type=str, default=config.get("resultspath", "./output/"),
                        help="Path to save results")
    # parser.add_argument("-im", "--images", action="store_true", help="Save visualizations")
    parser.add_argument("-lm", "--low_memory", action="store_true", help="Low memory flag")
    parser.add_argument("-n", "--nr_docs", type=int, default=config.get("nr_docs", 10),
                        help="Number of docs per topic to pass to representation model")
    parser.add_argument("-p", "--prompt", type=str, default=config.get("prompt", ""), help="Prompt to pass to LLM")
    parser.add_argument("-sz", "--min_topic_size", type=int, default=config.get("min_topic_size", 100),
                        help="Minimum topic size")

    # bertopic options

    parser.add_argument("-cl", "--clustering", type=str, choices=['hbdscan', 'kmeans', 'agglomerative'],
                        default=config.get("clustering_model", "hdbscan"), help="Clustering algorithm")
    # parser.add_argument("-llm", "--llm", type=str, choices=['gpt4o', 'mistral-small'], help="Low memory flag")
    parser.add_argument("-dim", "--dim_reduction", type=str, choices=['pca', 'umap'],
                        default=config.get("dimensionality_reduction", "umap"),
                        help="Dimensionality reduction algorithm")
    parser.add_argument("-s", "--save_embeddings", action="store_true", default=config.get("save_embeddings", False),
                        help="Save embeddings")

    args = parser.parse_args()

    print("Reading data...")
    documents = read_csv_files_in_directory(args.datapath)
    print("Done!")

    representation_model = MistralRepresentation(nr_docs=args.nr_docs, diversity=DOCUMENT_DIVERSITY, api="generate")
    medical_embedding_model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')
    custom_stop_words = {'with', 'my', 'your', 'she', 'this', 'was', 'her', 'have', 'as', 'he', 'him', 'but', 'not',
                         'so', 'are', 'at', 'be', 'has', 'do', 'got', 'how', 'on', 'or', 'would', 'will', 'what',
                         'they', 'if', 'or', 'get', 'can', 'we', 'me', 'can', 'has', 'his', 'there', 'them', 'just',
                         'am', 'by', 'that', 'from', 'it', 'is', 'in', 'you', 'also', 'very', 'had', 'a', 'an',
                         'for'}.union(stop_words.STOP_WORDS)

    vectorizer_model = CountVectorizer(stop_words=custom_stop_words)
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

    bertopic_model = BERTopicModel(ctfidf_model=ctfidf_model, embedding_model=medical_embedding_model, verbose=True,
                                   min_topic_size=args.min_topic_size, vectorizer_model=vectorizer_model,
                                   representation_model=representation_model, low_memory=args.low_memory)

    if os.path.exists(args.embeddings):
        print("Loading embeddings...")
    else:
        print("Generating embeddings...")

    document_embeddings = pickle.load(open(args.embeddings, "rb")) if os.path.exists(
        args.embeddings) else medical_embedding_model.encode(documents, show_progress_bar=True)

    print("Done!")

    if args.save_embeddings:
        pickle.dump(document_embeddings, open(os.path.join(args.result_path, "embeddings.pkl"), "wb"))

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
    results_df.to_csv(os.path.join(args.result_path, "output.csv"), index=False)


if __name__ == '__main__':
    main()
