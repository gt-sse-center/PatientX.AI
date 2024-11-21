from abc import ABC, abstractmethod
from os import PathLike
from typing import Union

import pandas as pd


class ClusteringModelInterface(ABC):
    @abstractmethod
    def getClusters(self, datapath: PathLike) -> Union[dict, pd.DataFrame]:
        '''

        :param datapath: path to txt file holding documents separated by '\n'
        :return: Union[dict, pd.DataFrame]: return results of topic modeling
        '''
        pass

    @abstractmethod
    def getModelType(self) -> str:
        '''
        :return: string describing model type
        '''
        pass
