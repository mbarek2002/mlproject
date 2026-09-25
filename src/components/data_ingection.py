import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import mlflow

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")
    source_data_path: str=os.path.join('notebooks','data','stud.csv')
    test_size: float=0.2
    random_state: int=42

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv(self.ingestion_config.source_data_path)
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(
                df,
                test_size=self.ingestion_config.test_size,
                random_state=self.ingestion_config.random_state
            )

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            if mlflow.active_run():
                mlflow.log_params({
                    "data_source": self.ingestion_config.source_data_path,
                    "n_rows": len(df),
                    "n_train": len(train_set),
                    "n_test": len(test_set),
                    "test_size": self.ingestion_config.test_size,
                    "random_state": self.ingestion_config.random_state,
                })

            logging.info("Inmgestion of the data iss completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    from src.pipeline.train_pipeline import TrainPipeline

    print(TrainPipeline().run())
