import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import structlog
import pandas as pd
from utils.utils import load_config_file

logger = structlog.getLogger()

class DataLoad:
    """Classe responsável pelo carregamento dos dados"""
    
    def __init__(self) -> None:
        pass
    
    def load_data(self, dataset_name: str) -> pd.DataFrame:
        """Retorna o DataFrame a partir do nome do arquivo fornecido.
        
            Args:
                dataset_name (str) : nome do arquivo

            return:
                pandas.DataFrame

        """
        try:
            dataset = load_config_file().get(dataset_name)
            if dataset is None:
                raise ValueError(f"O nome do arquivo fornecido está incorreto: {dataset}")
            dataframe = pd.read_csv(f'../data/raw/{dataset}')[load_config_file().get('columns_to_use')]
            return dataframe
        except ValueError as ve:
            logger.error(str(ve))
        except Exception as e:
            logger.error(str(e))