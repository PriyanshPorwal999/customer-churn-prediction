from src.utils.utils import save_object, load_object

sample_data = {
    "name": "Priyansh",
    "project": "Customer Churn"
}

save_object(
    "artifacts/test.pkl",
    sample_data
)

loaded_data = load_object(
    "artifacts/test.pkl"
)

print(loaded_data)