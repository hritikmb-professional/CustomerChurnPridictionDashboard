import pandas as pd

df = pd.read_csv("dataset.csv")
df.drop("customerID", axis=1, inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df.dropna(subset=["TotalCharges"], inplace=True)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)
categorical_cols = df.select_dtypes(include='object').columns.tolist()
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
df.to_csv("cleaned_dataset.csv", index=False)

print("Cleaned telecom churn dataset saved as 'cleaned_dataset.csv'")
