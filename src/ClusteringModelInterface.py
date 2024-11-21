from abc import ABC, abstractmethod


class ClusteringModelInterface(ABC):
    @abstractmethod
    def getClusters(self, documents):
        pass

    @abstractmethod
    def getModelType(self):
        pass
