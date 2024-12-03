from typing import List, Union

import numpy as np

from ClusteringModelInterface import ClusteringModelInterface
from bertopic import BERTopic
from typing_extensions import override
from pathlib import Path


class BERTopicModel(ClusteringModelInterface, BERTopic):
    def __init__(self, language='english', top_n_words=10, n_gram_range=(1, 1), min_topic_size=10, nr_topics=None,
                 low_memory=False, calculate_probabilities=False, seed_topic_list=None, zeroshot_topic_list=None,
                 zeroshot_min_similarity=0.7, embedding_model=None, umap_model=None, hdbscan_model=None,
                 vectorizer_model=None, ctfidf_model=None, representation_model=None, verbose=False):
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
