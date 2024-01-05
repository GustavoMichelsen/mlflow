import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import pandera
import structlog
import pandas as pd
from pandera import DataFrameSchema, Check, Column

logger = structlog.getLogger()

from utils.utils import load_config_file

class DataValidation:
    """Classe responsável pela validação dos dados"""
    
    def __init__(self) -> None:
        self.columns_to_use = load_config_file().get('columns_to_use')
        
    def check_data_shape(self, dataframe: pd.DataFrame) -> bool:
        """Checa se todas as colunas estão presentes no DataFrame.
        
            Args:
                Nenhum

            return:
                Bool
        """
        try:
            dataframe.columns = self.columns_to_use
            return True
        except Exception as e:
            logger.error(f'A Validação Falhou: {e}')
            return False
    
    def check_columns(self, dataframe: pd.DataFrame) -> bool:
        """Checa se os tipos de dados são os esperados.
        
            Args:
                dataframe : pandas.DataFrame

            return:
                Bool
        """
        schema = DataFrameSchema(
            {
            "target": Column(int, Check.isin([0,1 ]), Check(lambda x: x>0), coerce=True),
            "TaxaDeUtilizacaoDeLinhasNaoGarantidas": Column(float, nullable=True),
            "Idade": Column(int, nullable=True),
            "NumeroDeVezes30-59DiasAtrasoNaoPior": Column(int, nullable=True),
            "TaxaDeEndividamento": Column(float, nullable=True),
            "RendaMensal": Column(float, nullable=True),
            "NumeroDeLinhasDeCreditoEEmprestimosAbertos": Column(int, nullable=True),
            "NumeroDeVezes90DiasAtraso": Column(int, nullable=True),
            "NumeroDeEmprestimosOuLinhasImobiliarias": Column(int, nullable=True),
            "NumeroDeVezes60-89DiasAtrasoNaoPior": Column(int, nullable=True),
            "NumeroDeDependentes": Column(float, nullable=True)
            }
        )
        try:
            schema.validate(dataframe)
            return True
        except Exception as e:
            logger.error(f"Validação das colunas falhou: {e}")
            return False
    def run(self, dataframe: pd.DataFrame) -> bool:
        if self.check_data_shape(dataframe) and  self.check_columns(dataframe):
            logger.info(f'Validação Concluída')
            return True
        else:
            logger.error('Validação falhou')
            return False