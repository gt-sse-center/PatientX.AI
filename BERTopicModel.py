from ClusteringModel import ClusteringModel
from bertopic import BERTopic

class BERTopicModel(ClusteringModel, BERTopic):
    def __init__(self, embedding_model=None, dimensionality_reduction=None, cluster=None, tokenizer=None, weighting_scheme=None, representation_tuning=None, min_topic_size):
        self.embedding_model = embedding_model
        self.dimensionality_reduction = dimensionality_reduction
        self.cluster = cluster
        self.tokenizer = tokenizer
        self.weighting_scheme = weighting_scheme
        self.representation_tuning = representation_tuning
        self.min_topic_size = min_topic_size

        self.model = super(self.__class__, self).__init__(embedding_model=self.embedding_model, dimensionality_reduction=self.dimensionality_reduction, clustering_model=self.cluster, tokenizer=self.tokenizer, weighting_scheme=self.weighting_scheme, representation_model=self.representation_tuning, min_topic_size=self.min_topic_size)


    def getClusters(self, data):
        self.model.fit(data)

        return self.model.get_topic_info()

    def getModelType(self):
        return "BertTopicModel"

    def visualizeModel(self):
        self.model.visualize_topics()


