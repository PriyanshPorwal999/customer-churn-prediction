from src.components.data_transformation import DataTransformation

obj = DataTransformation()

preprocessor = obj.get_data_transformer_object()

print(type(preprocessor))