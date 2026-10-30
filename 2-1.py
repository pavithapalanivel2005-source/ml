
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv(r"C:\ml\Iris.csv")

print(df.head())
print(df.columns)

X = df[["SepalLengthCm", "SepalWidthCm", "PetalWidthCm"]]
y = df["PetalLengthCm"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Multiple Linear Regression")
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Petal Length")
plt.ylabel("Predicted Petal Length")
plt.title("Multiple Linear Regression")
plt.show()
--------------------------------
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load Iris Dataset
df = pd.read_csv(r"C:\ml\Iris.csv")

print(df.head())

# Input and Output
X = df[["SepalLengthCm"]]
y = df["PetalLengthCm"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Output
print("Simple Linear Regression")
print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_[0])
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Graph
plt.scatter(X_test, y_test, label="Actual")
plt.scatter(X_test, y_pred, color="red", label="Predicted")
plt.plot(X_test, y_pred, color="red")

plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.title("Iris Linear Regression")
plt.legend()
plt.show()