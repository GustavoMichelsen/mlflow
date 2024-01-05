import os
import sys

os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../models'))

import structlog
import pandas as pd
from sklearn.metrics         import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

logger = structlog.getLogger()

from utils.utils import load_config_file

class ModelEvaluation():
    """Objetivo da Classe: Fazer a avaliação do modelo"""
    def __init__(self,
                 model,
                 data_X : pd.DataFrame,
                 data_y : pd.DataFrame | list,
                 n_splits : int = 5):
        self.model = model
        self.data_X = data_X
        self.data_y = data_y
        self.n_splits = n_splits
    
    def cross_val_eval(self):
        """Faz a avaliação cruzada utilizando como métrica a rurva auc roc.
        
            Args:
                Nenhum

            return:
                list - Com os scores obtidos
        """
        sff = StratifiedKFold(n_splits=self.n_splits,
                              shuffle=True,
                              random_state=load_config_file().get('r_state'))
        score = cross_val_score(self.model,
                                X=self.data_X,
                                y=self.data_y,
                                cv=sff,
                                scoring='roc_auc')
        return score
    
    def roc_auc_eval(self, model, X: pd.DataFrame, y:pd.DataFrame | list):
        """Faz a avaliação do modelo com base na curva auc roc.
        
            Args:
                model: modelo Sklearn
                X : pandas.DataFrame - Dados de treino
                y : pandas.DataFrame | list - Dados de treino

            return:
                float - Score da curva auc roc
        """
        y_pred = model.predict_proba(X)[:, 1]
        return roc_auc_score(y, y_pred)
    
    @staticmethod
    def evaluate_prediction(y_true, y_pred_proba):
        """Faz a avaliação do modelo com base na curva auc roc.
        
            Args:
                X : pandas.DataFrame - Dados de treino
                y : pandas.DataFrame | list - Dados de treino

            return:
                float - Score da curva auc roc
        """
        return roc_auc_score(y_true, y_pred_proba)