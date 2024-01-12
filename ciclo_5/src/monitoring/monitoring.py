import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import sqlite3
import pandas as pd
from evidently.report        import Report
from evidently.metrics       import *
from evidently.metric_preset import DataDriftPreset
from evidently.test_preset   import DataDriftTestPreset

from data.dataload import DataLoad

class ModelMonitoring:
    def __init__(self) -> None:
        self.query = "SELECT * FROM predictions"

    def _get_pred_data(self):
        conn = sqlite3.connect('C:\Servidor\OneDrive - CALCADOS BEIRA RIO S A\Documentos\repos\mlflow\preds.db')
        df_pred = pd.read_sql_query(self.query, con=conn)
        conn.close()
        return df_pred
    
    def _get_training_data(self):
        dl = DataLoad()
        df_train - dl.load_data('train_dataset_name')
        return df_train

    def run(self):
        df_cur = self._get_pred_data()
        df_ref = self._get_training_data().drop(columns='target')

        model_card = Report(metrics=[DatasetSummaryMetric(), 
                                     DataDriftPreset(),
                                     DatasetMissingValuesMetric(),
                                     ])
        model_card.run(reference_data=df_ref, 
                       current_data=df_cur)

        model_card.save_html("../../data/docs/model_monitoring_report.html")

mm = ModelMonitoring()
mm.run()
