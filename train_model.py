import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

df = pd.read_csv("cleaned_dataset.csv")
X = df.drop("Churn", axis=1)
y = df["Churn"]
joblib.dump(X.columns.tolist(), "model_features.pkl")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))
joblib.dump(model, "churn_model.pkl")
print("Model and feature list saved successfully.")
