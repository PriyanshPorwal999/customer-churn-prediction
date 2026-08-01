from src.components.data_ingestion import DataIngestion

obj = DataIngestion()

print(obj.ingestion_config.train_data_path)
print(obj.ingestion_config.test_data_path)
print(obj.ingestion_config.raw_data_path)