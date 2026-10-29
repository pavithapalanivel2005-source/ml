import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv(r"C:\ml\Iris.csv")
X = df.drop(columns=["Species", "Id"], errors="ignore")
y = df["Species"]

a,b,c,d = train_test_split(X,y,test_size=0.2,random_state=42)
m = RandomForestClassifier(n_estimators=100,random_state=42)
m.fit(a,c)
p = m.predict(b)

cm = confusion_matrix(d,p)
print("Accuracy:",accuracy_score(d,p))
print("Confusion Matrix:\n",cm)

plt.imshow(cm)
plt.title("Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.show()