
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv(r"C:\ml\Iris.csv")

# Remove ID column if present
if "Id" in data.columns:
    data = data.drop("Id", axis=1)

# Input and output
X = data.drop("Species", axis=1)
y = data["Species"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=32
)

# KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

print("KNN Accuracy:",
      accuracy_score(y_test, knn.predict(X_test)))

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)

print("Gaussian NB Accuracy:",
      accuracy_score(y_test, y_pred))

# Encode class labels
le = LabelEncoder()
le.fit(y)

actual = le.transform(y_test)
predicted = le.transform(y_pred)

# Plot
plt.figure(figsize=(6, 6))
plt.scatter(actual, predicted, color="blue", s=60)
plt.plot([0, 2], [0, 2], "r--")

plt.xticks([0, 1, 2], le.classes_)
plt.yticks([0, 1, 2], le.classes_)

plt.xlabel("Actual Class")
plt.ylabel("Predicted Class")
plt.title("Naive Bayes - Actual vs Predicted")
plt.grid(True)
plt.tight_layout()
plt.show()
