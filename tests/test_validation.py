from src.components.data_validation import DataValidation

obj = DataValidation()

status = obj.validate_dataset()

print(status)