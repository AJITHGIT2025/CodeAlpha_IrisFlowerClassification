import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("Iris.csv")

print("First 5 Rows of Dataset:")
print(df.head())

X = df.drop(["Id", "Species"], axis=1)
y = df["Species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nModel Evaluation")
print("-" * 40)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


plt.figure(figsize=(8, 6))

species = df["Species"].unique()

for sp in species:
    subset = df[df["Species"] == sp]
    plt.scatter(
        subset["SepalLengthCm"],
        subset["PetalLengthCm"],
        label=sp
    )

plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.title("Iris Dataset Visualization")
plt.legend()
plt.grid(True)

plt.show()
