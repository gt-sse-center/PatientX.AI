from abc import ABC, abstractmethod
from os import PathLike


class ClusteringModelInterface(ABC):
    @abstractmethod
    def getClusters(self, datapath: PathLike):
        '''

        :param datapath: path to txt file holding documents separated by '\n'
        :return: topic modeling results
        '''
        pass

    @abstractmethod
    def getModelType(self):
        '''
        :return: string describing model type
        '''
        pass
