from dataclasses import dataclass
import os

import sys
import pandas as pd
from src.exception.custom_exception import CustomException
from src.logger.logger import logging


@dataclass
class DataValidationConfig:
    validation_status_file_path: str = os.path.join(
        "artifacts",
        "validation_status.txt"
    )


class DataValidation:
    def __init__(self):
        self.validation_config = DataValidationConfig()

    def validate_dataset(self):
        try: 
            df = pd.read_csv(
                os.path.join(
                    "artifacts",
                    "raw.csv"
                )
            )
            logging.info("Raw dataset loaded successfully")
            
            if len(df.columns) == 21:
                validation_status = True
            else:
                validation_status = False

            with open(
                self.validation_config.validation_status_file_path,
                "w"
            ) as f:
                f.write(
                    f"Validation Status: {validation_status}"
                )
            
            logging.info(f"Validation status: {validation_status}")
        
            return validation_status
        except Exception as e:
            raise CustomException(e, sys)
        