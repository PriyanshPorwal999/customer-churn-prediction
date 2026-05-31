from dataclasses import dataclass
import os


import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.exception.custom_exception import CustomException
from src.logger.logger import logging



@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")

    test_data_path: str = os.path.join("artifacts", "test.csv")

    raw_data_path: str = os.path.join("artifacts", "raw.csv")

    source_data_path: str = os.path.join(
        "data",
        "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    
    def initiate_data_ingestion(self):
        try:
            logging.info("Entered the data ingestion method")
            
            df = pd.read_csv(
                self.ingestion_config.source_data_path
            )

            logging.info("Data ingestion executed successfully")

            os.makedirs(
                os.path.dirname(
                    self.ingestion_config.train_data_path
                ),
                exist_ok=True
            )

            logging.info("Directory created, if not exist")

            df.to_csv(
                self.ingestion_config.raw_data_path,
                index=False
            )

            logging.info("Raw dataset saved")

            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )

            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False
            )

            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False
            )

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


