import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

df = pd.read_csv(r"C:\ml\Iris.csv")

print(df.head())

X = df.drop(['Species', 'Id'], axis=1, errors='ignore')
y = df['Species']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_predicted = model.predict(X_test)

print("Accuracy:", model.score(X_test, y_test))
print(classification_report(y_test, y_predicted))

plt.scatter(df.iloc[:, 0], df.iloc[:, 1],
            color='red', marker='+')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Iris Dataset')
plt.show()