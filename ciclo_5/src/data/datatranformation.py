import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import pandas as pd
from sklearn.model_selection import train_test_split

from utils.utils import load_config_file

class DataTransformation:
    """Classe responsável pela transformação dos dados"""
    def __init__(self, 
                 dataframe: pd.DataFrame):
        self.dataframe = dataframe
        self.target_name = load_config_file().get('target_name')
    
    def train_test_spliting(self):
        """Separa os dados em Treino e Validação.
        
            Args:
                Nenhum

            return:
                X_train : pandas.DataFrame
                x_val : pandas.DataFrame
                y_train : pandas.DataFrame
                y_val : pandas.DataFrame
        """
        X = self.dataframe.drop(columns=self.target_name)
        y = self.dataframe[self.target_name].values
        
        X_train, X_val, y_train, y_val = train_test_split(X, 
                                                          y, 
                                                          test_size = load_config_file().get('t_size'),
                                                          random_state= load_config_file().get('r_state'), 
                                                          stratify=y)
        
        return X_train, X_val, y_train, y_val