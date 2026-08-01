import pandas as pd

df = pd.read_csv("artifacts/raw.csv")

print(df["TotalCharges"].dtype)

print((df["TotalCharges"] == " ").sum())

print("\nShape:")
print(df.shape)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nUnique Values:")
print(df.nunique())