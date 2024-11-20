from abc import ABC, abstractmethod


class ClusteringModel(ABC):
    @abstractmethod
    def getClusters(self, data):
        pass

    @abstractmethod
    def visualizeModel(self):
        pass

    @abstractmethod
    def getModelType(self):
        pass