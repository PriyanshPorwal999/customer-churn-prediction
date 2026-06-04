from dataclasses import dataclass
import os

import os
import sys

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from src.exception.custom_exception import CustomException
from src.logger.logger import logging

from src.utils.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join(
        "artifacts",
        "preprocessor.pkl"
    )

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = (
            DataTransformationConfig()
        )

    def get_data_transformer_object(self):
        try: 
            numerical_columns = [
                "tenure",
                "MonthlyCharges",
                "TotalCharges"
            ]

            categorical_columns = [
                "gender",
                "SeniorCitizen",
                "Partner",
                "Dependents",
                "PhoneService",
                "MultipleLines",
                "InternetService",
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
                "Contract",
                "PaperlessBilling",
                "PaymentMethod"
            ]

            num_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="median")
                    ),

                    (
                        "scaler",
                        StandardScaler()
                    )
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        )
                    ),

                    (
                        "one_hot_encoder",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        )
                    )
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    (
                        "num_pipeline",
                        num_pipeline,
                        numerical_columns
                    ),

                    (
                        "cat_pipeline",
                        cat_pipeline,
                        categorical_columns
                    )
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_transformation(
        self,
        train_path,
        test_path
    ):
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        train_df["TotalCharges"] = pd.to_numeric(
            train_df["TotalCharges"],
            errors="coerce"
        )

        test_df["TotalCharges"] = pd.to_numeric(
            test_df["TotalCharges"],
            errors="coerce"
        )

        target_column_name = "Churn"
        preprocessing_obj = (
            self.get_data_transformer_object()
        )

        input_feature_train_df = train_df.drop(
            columns=[target_column_name, "customerID"],
            axis=1
        )

        input_feature_test_df = test_df.drop(
            columns=[target_column_name, "customerID"],
            axis=1
        )

        input_feature_train_arr = preprocessing_obj.fit_transform(
            input_feature_train_df
        )

        input_feature_test_arr = preprocessing_obj.transform(
            input_feature_test_df
        )

        target_feature_train_df = (
            train_df[target_column_name]
        ).map({"No": 0, "Yes": 1})

        target_feature_test_df = (
            test_df[target_column_name]
        ).map({"No": 0, "Yes": 1})

        # target_feature_train_df = (
        #     train_df[target_column_name],
        #     target_feature_train_df
        #     .map({"No": 0, "Yes": 1})
        # )

        # target_feature_test_df = (
        #     test_df[target_column_name],
        #     target_feature_test_df
        #     .map({"No": 0, "Yes": 1})
        # )

        train_arr = np.c_[
            input_feature_train_arr,
            np.array(target_feature_train_df)
        ]

        test_arr = np.c_[
            input_feature_test_arr,
            np.array(target_feature_test_df)
        ]

        save_object(
            file_path=self.data_transformation_config.preprocessor_obj_file_path,
            obj=preprocessing_obj
        )

        return (
            train_arr,
            test_arr,
            self.data_transformation_config.preprocessor_obj_file_path
        )
    