import os

import yaml
import joblib

def load_config_file():
    """Faz a avaliação do modelo com base na curva auc roc.
        
            Args:
                Nenhum 

            return:
                dicionário {} - Contem todos os argumentos de configuração
        """
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    caminho_relativo = os.path.join('..', '..', 'config', 'config.yaml')

    config_path = os.path.abspath(os.path.join(diretorio_atual, caminho_relativo))

    config_file = yaml.safe_load(open(config_path, 'rb'))

    return config_file

def save_model(model):
    """Salva o modelo.
        
            Args:
                model: modelo Sklearn

            return:
                Nenhum
        """
    model_name = load_config_file().get('model_name')
    model_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'models', model_name))
    joblib.dump(model, model_path)
    