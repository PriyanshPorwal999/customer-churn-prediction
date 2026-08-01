from src.components.data_ingestion import DataIngestion

obj = DataIngestion()

train_path, test_path = obj.initiate_data_ingestion()

print(train_path)
print(test_path)

print("Data ingestion executed successfully")