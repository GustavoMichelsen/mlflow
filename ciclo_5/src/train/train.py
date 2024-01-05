import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import joblib
import pandas as pd

from utils.utils import load_config_file



class TrainModel:
    """Objetivo da Classe: Fazer o treinameto de um modelo"""
    def __init__(self, 
                 dados_x: pd.DataFrame,
                 dados_y: pd.DataFrame):
        """Args:
                dados_x : pandas.DataFrame - Dados de treino
                dadox_y : pandas.DataFrame | list - Dados de Treino
        """
        self.dados_x = dados_x
        self.dados_y = dados_y
        self.model_name = load_config_file().get('model_name')
        
    def train(self, model):
        """Efetua o treinameto do modelo e salva uma cópia 
        no diretório em que o arquivo está sendo executado.
        
            Args:
                model: modelo Sklearn

            return:
                Modelo treinado
        """
        model.fit(self.dados_x, self.dados_y)
        joblib.dump(model, self.model_name)
        return model
