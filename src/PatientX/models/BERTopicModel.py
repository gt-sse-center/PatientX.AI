from typing import List, Union, Mapping, Tuple, Any

import numpy as np
import pandas as pd

from PatientX.models.ClusteringModelInterface import ClusteringModelInterface
from bertopic import BERTopic
from bertopic.cluster import BaseCluster
from bertopic.backend._utils import select_backend
from bertopic._utils import (
    MyLogger,
    check_documents_type,
    check_embeddings_shape,
)
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from bertopic.representation import BaseRepresentation
from typing_extensions import override
from pathlib import Path

logger = MyLogger()


class BERTopicModel(ClusteringModelInterface, BERTopic):
    def __init__(self, nr_representative_docs, language='english', top_n_words=10, n_gram_range=(1, 1), min_topic_size=10, nr_topics=None,
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
        self.fit_documents = None
        self.nr_representative_docs = nr_representative_docs

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
            nr_repr_docs=self.nr_representative_docs,
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
    def fit(self,
        documents: List[str],
        embeddings: np.ndarray = None,
        images: List[str] = None,
        y: Union[List[int], np.ndarray] = None,):
        if documents is not None:
            check_documents_type(documents)
            check_embeddings_shape(embeddings, documents)

        doc_ids = range(len(documents)) if documents is not None else range(len(images))
        documents = pd.DataFrame({"Document": documents, "ID": doc_ids, "Topic": None, "Image": images})

        # Extract embeddings
        if embeddings is None:
            logger.info("Embedding - Transforming documents to embeddings.")
            self.embedding_model = select_backend(self.embedding_model, language=self.language, verbose=self.verbose)
            embeddings = self._extract_embeddings(
                documents.Document.values.tolist(),
                images=images,
                method="document",
                verbose=self.verbose,
            )
            logger.info("Embedding - Completed \u2713")
        else:
            if self.embedding_model is not None:
                self.embedding_model = select_backend(
                    self.embedding_model, language=self.language, verbose=self.verbose
                )

        # Guided Topic Modeling
        if self.seed_topic_list is not None and self.embedding_model is not None:
            y, embeddings = self._guided_topic_modeling(embeddings)

        # Reduce dimensionality and fit UMAP model
        umap_embeddings = self._reduce_dimensionality(embeddings, y)

        # Zero-shot Topic Modeling
        if self._is_zeroshot():
            documents, embeddings, assigned_documents, assigned_embeddings = self._zeroshot_topic_modeling(
                documents, embeddings
            )

            # Filter UMAP embeddings to only non-assigned embeddings to be used for clustering
            if len(documents) > 0:
                umap_embeddings = self.umap_model.transform(embeddings)

        if len(documents) > 0:
            # Cluster reduced embeddings
            documents, probabilities = self._cluster_embeddings(umap_embeddings, documents, y=y)
            if self._is_zeroshot() and len(assigned_documents) > 0:
                documents, embeddings = self._combine_zeroshot_topics(
                    documents, embeddings, assigned_documents, assigned_embeddings
                )
        else:
            # All documents matches zero-shot topics
            documents = assigned_documents
            embeddings = assigned_embeddings

        # Sort and Map Topic IDs by their frequency
        if not self.nr_topics:
            documents = self._sort_mappings_by_frequency(documents)

        # Create documents from images if we have images only
        if documents.Document.values[0] is None:
            custom_documents = self._images_to_text(documents, embeddings)

            # Extract topics by calculating c-TF-IDF
            self._extract_topics(custom_documents, embeddings=embeddings)
            self._create_topic_vectors(documents=documents, embeddings=embeddings)

            # Reduce topics
            if self.nr_topics:
                custom_documents = self._reduce_topics(custom_documents)

            # Save the top 3 most representative documents per topic
            self._save_representative_docs(custom_documents)
        else:
            # Extract topics by calculating c-TF-IDF
            self._extract_topics(documents, embeddings=embeddings, verbose=self.verbose)

            # Reduce topics
            if self.nr_topics:
                documents = self._reduce_topics(documents)

            # Save the top 3 most representative documents per topic
            self._save_representative_docs(documents)

        # In the case of zero-shot topics, probability will come from cosine similarity,
        # and the HDBSCAN model will be removed
        if self._is_zeroshot() and len(assigned_documents) > 0:
            self.hdbscan_model = BaseCluster()
            sim_matrix = cosine_similarity(embeddings, np.array(self.topic_embeddings_))

            if self.calculate_probabilities:
                self.probabilities_ = sim_matrix
            else:
                self.probabilities_ = np.max(sim_matrix, axis=1)
        else:
            self.probabilities_ = self._map_probabilities(probabilities, original_topics=True)
        predictions = documents.Topic.to_list()

        # NOTE: this line is the only change from bertopic.BERTopic.fit_transform()
        self.fit_documents = documents

        return predictions, self.probabilities_

    def get_bertopic_only_results(self) -> tuple[Any, dict[int, list[tuple[str | list[str], Any] | tuple[str, float]]]]:
        return self.representative_docs_, self.bertopic_representative_words

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
