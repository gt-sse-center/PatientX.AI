from typing import List, Union, Mapping, Tuple, Any, Dict

import numpy as np
import pandas as pd
from pandas import DataFrame

from PatientX.models.ClusteringModelInterface import ClusteringModelInterface
from bertopic import BERTopic
from bertopic._utils import (
    MyLogger,
    check_documents_type,
    check_embeddings_shape,
    check_is_fitted,
)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.cluster import BaseCluster
from bertopic.representation import BaseRepresentation
from bertopic.backend._utils import select_backend
from typing_extensions import override
from pathlib import Path

logger = MyLogger()


class BERTopicModel(ClusteringModelInterface, BERTopic):
    def __init__(self, language='english', top_n_words=10, n_gram_range=(1, 1), min_topic_size=10, nr_topics=None,
                 low_memory=False, calculate_probabilities=False, seed_topic_list=None, zeroshot_topic_list=None,
                 zeroshot_min_similarity=0.7, embedding_model=None, umap_model=None, hdbscan_model=None,
                 vectorizer_model=None, ctfidf_model=None, representation_model=None, verbose=False, nr_docs=10):
        self.language = language
        self.top_n_words = top_n_words
        self.n_gram_range = n_gram_range
        self.min_topic_size = min_topic_size
        self.nr_topics = nr_topics
        self.low_memory = low_memory
        self.calculate_probabilities = calculate_probabilities
        self.seed_topic_list = seed_topic_list
        self.zeroshot_topic_list = zeroshot_topic_list
        self.zeroshot_min_similarity = zeroshot_min_similarity
        self.embedding_model = embedding_model
        self.umap_model = umap_model
        self.hdbscan_model = hdbscan_model
        self.vectorizer_model = vectorizer_model
        self.ctfidf_model = ctfidf_model
        self.representation_model = representation_model
        self.verbose = verbose
        self.nr_docs = nr_docs

        super().__init__(language=language, top_n_words=top_n_words,
                         n_gram_range=n_gram_range, min_topic_size=min_topic_size,
                         nr_topics=nr_topics, low_memory=low_memory,
                         calculate_probabilities=calculate_probabilities,
                         seed_topic_list=seed_topic_list,
                         zeroshot_topic_list=zeroshot_topic_list,
                         zeroshot_min_similarity=zeroshot_min_similarity,
                         embedding_model=embedding_model, umap_model=umap_model,
                         hdbscan_model=hdbscan_model,
                         vectorizer_model=vectorizer_model, ctfidf_model=ctfidf_model,
                         representation_model=representation_model, verbose=verbose)

    @override
    def _save_representative_docs(self, documents: pd.DataFrame):
        repr_docs, _, _, _ = self._extract_representative_docs(
            self.c_tf_idf_,
            documents,
            self.topic_representations_,
            nr_samples=500,
            nr_repr_docs=self.nr_docs,
        )
        self.representative_docs_ = repr_docs

    @override
    def _extract_topics(
        self,
        documents: pd.DataFrame,
        embeddings: np.ndarray = None,
        mappings=None,
        verbose: bool = False,
    ):
        """Extract topics from the clusters using a class-based TF-IDF.

        Arguments:
            documents: Dataframe with documents and their corresponding IDs
            embeddings: The document embeddings
            mappings: The mappings from topic to word
            verbose: Whether to log the process of extracting topics

        Returns:
            c_tf_idf: The resulting matrix giving a value (importance score) for each word per topic
        """
        if verbose:
            logger.info("Representation - Extracting topics from clusters using representation models.")
        print("Documents DF passed in: ")
        print(documents.head(10))
        documents_per_topic = documents.groupby(["Topic"], as_index=False).agg({"Document": " ".join})
        self.c_tf_idf_, words = self._c_tf_idf(documents_per_topic)
        self.topic_representations_ = self._extract_words_per_topic(words, documents)
        self._create_topic_vectors(documents=documents, embeddings=embeddings, mappings=mappings)
        self.bertopic_only_results = documents_per_topic.copy(deep=True)
        if verbose:
            logger.info("Representation - Completed \u2713")

    @override
    def getClusters(self, datapath):
        if Path(datapath).is_file():
            with open(datapath, 'r', encoding='utf-8') as file:
                documents = file.read().split('\n')  # Split on newline to get individual documents

            super().fit_transform(documents)

            return super().get_topic_info()
        else:
            raise FileNotFoundError("The specified datapath does not exist")

    @override
    def getModelType(self):
        return "BERTopic"

    def visualizeModel(self):
        """
        Visualize topics, dendogram, word occurences bar chart, term score decline
        :return: None
        """
        super().visualize_topics()
        # visualize hierarchy
        super().visualize_hierarchy()

        # visualize topic word scores
        super().visualize_barchart()

        # visualize term rank
        super().visualize_term_rank()

    def get_bertopic_only_results(self) -> tuple[
        Any, Any, dict[int, list[tuple[str | list[str], Any] | tuple[str, float]]]]:
        return (self.bertopic_only_results, self.representative_docs_, self.bertopic_representative_words)

    @override
    def _extract_words_per_topic(
        self,
        words: List[str],
        documents: pd.DataFrame,
        c_tf_idf: csr_matrix = None,
        calculate_aspects: bool = True,
    ) -> Mapping[str, List[Tuple[str, float]]]:
        """
        NOTE: this function overrides bertopic._extract_words_per_topic()
        The only difference is that we explicitly save the representative words to self.bertopic_representative_words
        so that we can later save the intermediate bertopic results


        Based on tf_idf scores per topic, extract the top n words per topic.

        If the top words per topic need to be extracted, then only the `words` parameter
        needs to be passed. If the top words per topic in a specific timestamp, then it
        is important to pass the timestamp-based c-TF-IDF matrix and its corresponding
        labels.

        Arguments:
            words: List of all words (sorted according to tf_idf matrix position)
            documents: DataFrame with documents and their topic IDs
            c_tf_idf: A c-TF-IDF matrix from which to calculate the top words
            calculate_aspects: Whether to calculate additional topic aspects

        Returns:
            topics: The top words per topic
        """
        if c_tf_idf is None:
            c_tf_idf = self.c_tf_idf_

        labels = sorted(list(documents.Topic.unique()))
        labels = [int(label) for label in labels]

        # Get at least the top 30 indices and values per row in a sparse c-TF-IDF matrix
        top_n_words = max(self.top_n_words, 30)
        indices = self._top_n_idx_sparse(c_tf_idf, top_n_words)
        scores = self._top_n_values_sparse(c_tf_idf, indices)
        sorted_indices = np.argsort(scores, 1)
        indices = np.take_along_axis(indices, sorted_indices, axis=1)
        scores = np.take_along_axis(scores, sorted_indices, axis=1)

        # Get top 30 words per topic based on c-TF-IDF score
        base_topics = {
            label: [
                (words[word_index], score) if word_index is not None and score > 0 else ("", 0.00001)
                for word_index, score in zip(indices[index][::-1], scores[index][::-1])
            ]
            for index, label in enumerate(labels)
        }

        # NOTE: this is the only change from bertopic._extract_words_per_topic()
        self.bertopic_representative_words = {label: values[: self.top_n_words] for label, values in base_topics.items()}

        # Fine-tune the topic representations
        topics = base_topics.copy()
        if not self.representation_model:
            # Default representation: c_tf_idf + top_n_words
            topics = {label: values[: self.top_n_words] for label, values in topics.items()}
        elif isinstance(self.representation_model, list):
            for tuner in self.representation_model:
                topics = tuner.extract_topics(self, documents, c_tf_idf, topics)
        elif isinstance(self.representation_model, BaseRepresentation):
            topics = self.representation_model.extract_topics(self, documents, c_tf_idf, topics)
        elif isinstance(self.representation_model, dict):
            if self.representation_model.get("Main"):
                main_model = self.representation_model["Main"]
                if isinstance(main_model, BaseRepresentation):
                    topics = main_model.extract_topics(self, documents, c_tf_idf, topics)
                elif isinstance(main_model, list):
                    for tuner in main_model:
                        topics = tuner.extract_topics(self, documents, c_tf_idf, topics)
                else:
                    raise TypeError(f"unsupported type {type(main_model).__name__} for representation_model['Main']")
            else:
                # Default representation: c_tf_idf + top_n_words
                topics = {label: values[: self.top_n_words] for label, values in topics.items()}
        else:
            raise TypeError(f"unsupported type {type(self.representation_model).__name__} for representation_model")

        # Extract additional topic aspects
        if calculate_aspects and isinstance(self.representation_model, dict):
            for aspect, aspect_model in self.representation_model.items():
                if aspect != "Main":
                    aspects = base_topics.copy()
                    if not aspect_model:
                        # Default representation: c_tf_idf + top_n_words
                        aspects = {label: values[: self.top_n_words] for label, values in aspects.items()}
                    if isinstance(aspect_model, list):
                        for tuner in aspect_model:
                            aspects = tuner.extract_topics(self, documents, c_tf_idf, aspects)
                    elif isinstance(aspect_model, BaseRepresentation):
                        aspects = aspect_model.extract_topics(self, documents, c_tf_idf, aspects)
                    else:
                        raise TypeError(
                            f"unsupported type {type(aspect_model).__name__} for representation_model[{repr(aspect)}]"
                        )
                    self.topic_aspects_[aspect] = aspects

        return topics

    def visualize_document_datamap(
            self,
            docs: List[str],
            topics: List[int] = None,
            embeddings: np.ndarray = None,
            reduced_embeddings: np.ndarray = None,
            custom_labels: Union[bool, str] = False,
            title: str = "Documents and Topics",
            sub_title: Union[str, None] = None,
            width: int = 1200,
            height: int = 1200,
            **datamap_kwds,
    ):
        """
        Display documents/clustering with datamapplot library
        
        :param docs: documents trained on
        :param topics: topics learned
        :param embeddings: embeddings of topics
        :param reduced_embeddings: embeddings passed through dimensionality reduction
        :param custom_labels: custom labels
        :param title: plot title
        :param sub_title: plot subtitle
        :param width: plot width
        :param height: plot height
        :param datamap_kwds: datamap
        :return: None
        """
        super().visualize_document_datamap(docs, topics, embeddings, reduced_embeddings, custom_labels, title,
                                           sub_title, width, height, **datamap_kwds)
