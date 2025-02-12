from typing import List, Union

import numpy as np
import pandas as pd

from PatientX.models.ClusteringModelInterface import ClusteringModelInterface
from bertopic import BERTopic
from bertopic._utils import (
    MyLogger,
    check_documents_type,
    check_embeddings_shape,
    check_is_fitted,
    validate_distance_matrix,
    select_topic_representation,
    get_unique_distances,
)
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from bertopic.vectorizers import ClassTfidfTransformer
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

    @override
    def update_topics(
            self,
            docs: List[str],
            images: List[str] = None,
            topics: List[int] = None,
            top_n_words: int = 10,
            n_gram_range: Tuple[int, int] = None,
            vectorizer_model: CountVectorizer = None,
            ctfidf_model: ClassTfidfTransformer = None,
            representation_model: BaseRepresentation = None,
    ):
        """Updates the topic representation by recalculating c-TF-IDF with the new
        parameters as defined in this function.

        When you have trained a model and viewed the topics and the words that represent them,
        you might not be satisfied with the representation. Perhaps you forgot to remove
        stop_words or you want to try out a different n_gram_range. This function allows you
        to update the topic representation after they have been formed.

        Arguments:
            docs: The documents you used when calling either `fit` or `fit_transform`
            images: The images you used when calling either `fit` or `fit_transform`
            topics: A list of topics where each topic is related to a document in `docs`.
                    Use this variable to change or map the topics.
                    NOTE: Using a custom list of topic assignments may lead to errors if
                          topic reduction techniques are used afterwards. Make sure that
                          manually assigning topics is the last step in the pipeline
            top_n_words: The number of words per topic to extract. Setting this
                         too high can negatively impact topic embeddings as topics
                         are typically best represented by at most 10 words.
            n_gram_range: The n-gram range for the CountVectorizer.
            vectorizer_model: Pass in your own CountVectorizer from scikit-learn
            ctfidf_model: Pass in your own c-TF-IDF model to update the representations
            representation_model: Pass in a model that fine-tunes the topic representations
                                  calculated through c-TF-IDF. Models from `bertopic.representation`
                                  are supported.

        Examples:
        In order to update the topic representation, you will need to first fit the topic
        model and extract topics from them. Based on these, you can update the representation:

        ```python
        topic_model.update_topics(docs, n_gram_range=(2, 3))
        ```

        You can also use a custom vectorizer to update the representation:

        ```python
        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english")
        topic_model.update_topics(docs, vectorizer_model=vectorizer_model)
        ```

        You can also use this function to change or map the topics to something else.
        You can update them as follows:

        ```python
        topic_model.update_topics(docs, my_updated_topics)
        ```
        """
        check_documents_type(docs)
        check_is_fitted(self)
        if not n_gram_range:
            n_gram_range = self.n_gram_range

        if top_n_words > 100:
            logger.warning(
                "Note that extracting more than 100 words from a sparse " "can slow down computation quite a bit."
            )
        self.top_n_words = top_n_words
        self.vectorizer_model = vectorizer_model or CountVectorizer(ngram_range=n_gram_range)
        self.ctfidf_model = ctfidf_model or ClassTfidfTransformer()
        self.representation_model = representation_model

        if topics is None:
            topics = self.topics_
        else:
            logger.warning(
                "Using a custom list of topic assignments may lead to errors if "
                "topic reduction techniques are used afterwards. Make sure that "
                "manually assigning topics is the last step in the pipeline."
                "Note that topic embeddings will also be created through weighted"
                "c-TF-IDF embeddings instead of centroid embeddings."
            )

        documents = pd.DataFrame({"Document": docs, "Topic": topics, "ID": range(len(docs)), "Image": images})
        documents_per_topic = documents.groupby(["Topic"], as_index=False).agg({"Document": " ".join})

        # Update topic sizes and assignments
        self._update_topic_size(documents)

        # Extract words and update topic labels
        self.c_tf_idf_, words = self._c_tf_idf(documents_per_topic)
        self.topic_representations_ = self._extract_words_per_topic(words, documents)

        # Update topic vectors
        if set(topics) != self.topics_:
            # Remove outlier topic embedding if all that has changed is the outlier class
            same_position = all(
                [
                    True if old_topic == new_topic else False
                    for old_topic, new_topic in zip(self.topics_, topics)
                    if old_topic != -1
                ]
            )
            if same_position and -1 not in topics and -1 in self.topics_:
                self.topic_embeddings_ = self.topic_embeddings_[1:]
            else:
                self._create_topic_vectors()



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
