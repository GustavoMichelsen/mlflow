import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import structlog
import pandas as pd
from sklearn.pipeline import Pipeline

logger = structlog.getLogger()

from utils.utils import load_config_file

class DataPreprocess:
    """Objetivo da Classe: Fazer o pre-processamento dos dados"""
    def __init__(self,
                 pipe: Pipeline):
        self.pipe = pipe
        self.treined_pipeline = None

    def train(self, dataframe: pd.DataFrame):
        """Realiza o treinameto do pipe conforme o DataFrame passado
        
            Args:
                dataframe (DataFrame) : pandas.DataFrame

            return:
                Pipeline treinado
        """
        self.treined_pipeline = self.pipe.fit(dataframe)
    
    def transform(self, dataframe: pd.DataFrame):
        """Faz as transformações no DataFrame
        
            Args:
                dataframe (DataFrame) : pandas.DataFrame

            return:
                pandas.DataFrame com as transformações aplicadas
        """
        if self.treined_pipeline is None:
            raise ValueError(f'Pipeline não foi treinado')
        data_processed = self.treined_pipeline.transform(dataframe)
        return data_processed