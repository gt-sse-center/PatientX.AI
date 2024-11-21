from abc import ABC, abstractmethod


class ClusteringModel(ABC):
    @abstractmethod
    def getClusters(self, documents):
        pass

    @abstractmethod
    def getModelType(self):
        pass
